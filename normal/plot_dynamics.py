"""PNG/SVG dynamics figures from actual normal-mode execution telemetry.

No measurements are synthesized, interpolated or smoothed. In particular,
missing/invalid SnapFT readings remain NaN rather than a false zero torque.
Successful validation exports require the new telemetry channels. Explicit
diagnostic exports of incomplete runs use a different filename and title.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
JOINT_COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9']
ARM_COLORS={'A':'#d67925','B':'#367fb6'}
VELOCITY_LIMIT=.60
SNAP_LIMIT=1.20


def _finite(value: Any) -> float:
    if isinstance(value,bool):return math.nan
    try:answer=float(value)
    except (TypeError,ValueError):return math.nan
    return answer if math.isfinite(answer) else math.nan


def _vector(row: Mapping, key: str) -> list[float]:
    value=row.get(key)
    if not isinstance(value,Sequence) or isinstance(value,(str,bytes)) or len(value)!=6:
        return [math.nan]*6
    return [_finite(v) for v in value]


def _style():
    font=Path(r'C:\Windows\Fonts\msyh.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        family=font_manager.FontProperties(fname=str(font)).get_name()
    else:family='DejaVu Sans'
    plt.rcParams.update({'font.family':family,'font.size':10,'axes.unicode_minus':False,
                         'svg.fonttype':'none','figure.facecolor':'white',
                         'savefig.facecolor':'white','axes.spines.top':False,
                         'axes.spines.right':False,'axes.titlepad':12,
                         'xtick.labelsize':9,'ytick.labelsize':9})


def _read(execution):
    if isinstance(execution,(str,Path)):
        source=Path(execution)
        return json.loads(source.read_text(encoding='utf-8')),str(source)
    if isinstance(execution,Mapping):return dict(execution),'execution.json（本次内存日志）'
    raise TypeError('execution must be a telemetry mapping or JSON path')


def _signals(execution,allow_incomplete):
    if execution.get('success') is not True and not allow_incomplete:
        raise ValueError('Execution is not successful; use allow_incomplete=True only for an explicitly labeled diagnostic figure.')
    trace=execution.get('trace',[])
    if not isinstance(trace,Sequence) or len(trace)<2:raise ValueError('At least two actual trace samples are required.')
    rows=[]
    for row in trace:
        if not isinstance(row,Mapping):raise ValueError('Trace rows must be mappings.')
        stamp=_finite(row.get('time',row.get('time_s')))
        if not math.isfinite(stamp):raise ValueError('Every trace row needs a finite simulation time.')
        rows.append((stamp,row))
    rows.sort(key=lambda item:item[0])
    t=np.array([v[0] for v in rows]);ordered=[v[1] for v in rows]
    if np.any(np.diff(t)<=0):raise ValueError('Actual trace times must be strictly increasing; duplicate timestamps are not discarded.')
    velocity={a:np.asarray([_vector(r,a+'_joint_velocity_rad_s') for r in ordered]) for a in ['A','B']}
    torque={a:np.asarray([_vector(r,a+'_joint_torque_Nm') for r in ordered]) for a in ['A','B']}
    reaction={a:np.asarray([_finite(r.get('support',{}).get(a)) if isinstance(r.get('support'),Mapping) else math.nan
                           for r in ordered]) for a in ['A','B']}
    load=np.asarray([_finite(r.get('applied_insertion_load_N')) for r in ordered])
    bond=[]
    for row in ordered:
        snap=row.get('snap',{})
        sensor=snap.get('beam',{}) if isinstance(snap,Mapping) else {}
        value=math.nan
        if isinstance(sensor,Mapping) and sensor.get('valid'):
            value=_finite(sensor.get('torque_norm_Nm'))
            if not math.isfinite(value):
                vector=sensor.get('torque_Nm')
                if isinstance(vector,Sequence) and len(vector)==3:
                    v=np.asarray([_finite(q) for q in vector])
                    if np.all(np.isfinite(v)):value=float(np.linalg.norm(v))
        bond.append(value)
    bond=np.asarray(bond)
    missing=[]
    for a in ['A','B']:
        if not np.all(np.isfinite(velocity[a])):missing.append(a+'关节速度')
        if not np.all(np.isfinite(torque[a])):missing.append(a+'关节电机力矩')
        if not np.all(np.isfinite(reaction[a])):missing.append(a+'支撑反力')
    if not np.all(np.isfinite(load)):missing.append('实际外加载荷')
    if not np.any(np.isfinite(bond)):missing.append('beam SnapFT 有效力矩')
    if missing and not allow_incomplete:
        raise ValueError('Incomplete telemetry cannot form a validation figure: '+', '.join(missing))
    return {'time':t,'rows':ordered,'velocity':velocity,'torque':torque,
            'reaction':reaction,'load':load,'bond':bond,'missing':missing}


def _intervals(t,mask):
    """Convert observed active samples to display intervals, not fake samples."""
    intervals=[];begin=None
    half=float(np.median(np.diff(t)))/2
    for i,active in enumerate(mask):
        if active and begin is None:begin=i
        if begin is not None and (not active or i==len(mask)-1):
            last=i if active else i-1
            intervals.append((max(t[0],t[begin]-half),min(t[-1],t[last]+half)))
            begin=None
    return intervals


def _force_window(data,execution):
    t=data['time'];active=np.isfinite(data['load']) & (data['load']>1e-9)
    if np.any(active):
        stamps=t[active]
        return (max(t[0],stamps[0]-2.),min(t[-1],stamps[-1]+2.)),'插装外载非零时段前后各 2 s'
    stamps=[]
    for event in execution.get('events',[]):
        if isinstance(event,Mapping) and 'support' in str(event.get('kind','')):
            stamp=_finite(event.get('time',event.get('time_s')))
            if math.isfinite(stamp):stamps.append(stamp)
    if stamps:return (max(t[0],min(stamps)-2.),min(t[-1],max(stamps)+2.)),'支撑事件时段前后各 2 s'
    reacted=np.zeros(len(t),dtype=bool)
    for values in data['reaction'].values():reacted|=np.isfinite(values)&(values>.15)
    if np.any(reacted):
        stamps=t[reacted]
        return (max(t[0],stamps[0]-2.),min(t[-1],stamps[-1]+2.)),'实测支撑反力时段前后各 2 s'
    return (t[0],t[-1]),'本日志无可识别插装时段，显示全程'


def _axes(ax,title,ylabel):
    ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
    ax.set_xlabel('仿真时间 / s');ax.set_ylabel(ylabel)
    ax.grid(True,color='#e6e8eb',linewidth=.55)
    ax.set_axisbelow(True);ax.xaxis.set_major_locator(MaxNLocator(7))


def plot_dynamics(execution: Mapping | str | Path, outdir: str | Path | None=None,
                  *,allow_incomplete: bool=False) -> list[str]:
    """Export four real telemetry panels; refuse unvalidated success claims.

    ``allow_incomplete`` is for inspecting a real failed/older log. Such plots
    are labeled diagnostic and use ``normal_dynamics_diagnostic`` filenames.
    The default successful output is ``normal_dynamics_validation.png/svg``.
    """
    record,source=_read(execution)
    data=_signals(record,allow_incomplete);_style()
    t=data['time'];window,zoom_note=_force_window(data,record)
    diagnostic=record.get('success') is not True or bool(data['missing'])
    outdir=Path(outdir or ROOT/'results'/'normal');outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(15.6,9.1))
    fig.subplots_adjust(left=.075,right=.975,bottom=.14,top=.82,wspace=.24,hspace=.45)
    title='正常模式动力学实测' if not diagnostic else '正常模式动力学诊断：该日志未形成成功验收'
    fig.suptitle(title,x=.075,y=.975,ha='left',fontsize=17,fontweight='bold')
    fig.text(.075,.937,f'来源：{Path(source).name}　|　实际仿真采样 {len(t)} 点　|　{t[0]:.2f}–{t[-1]:.2f} s　|　原始采样，未平滑',fontsize=10,color='#505b68')
    handles=[Line2D([0],[0],color=c,lw=2,label=f'关节 {i+1}') for i,c in enumerate(JOINT_COLORS)]
    handles += [Line2D([0],[0],color='#303846',lw=1.7,label='A 臂：实线'),
                Line2D([0],[0],color='#303846',lw=1.7,ls=(0,(5,2)),label='B 臂：虚线')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.068,.916),ncol=8,
               frameon=False,columnspacing=1.55,handlelength=2.,fontsize=9)
    velocity_ax,torque_ax,force_ax,bond_ax=axes.flatten()
    for arm in ['A','B']:
        style='-' if arm=='A' else (0,(5,2))
        for joint,color in enumerate(JOINT_COLORS):
            velocity_ax.plot(t,data['velocity'][arm][:,joint],color=color,lw=.95,ls=style,alpha=.88)
            torque_ax.plot(t,data['torque'][arm][:,joint],color=color,lw=.95,ls=style,alpha=.88)
    for limit in [-VELOCITY_LIMIT,VELOCITY_LIMIT]:
        velocity_ax.axhline(limit,color='#262b32',lw=1.05,ls=(0,(2,3)))
    _axes(velocity_ax,'（a）双臂实际关节速度','角速度 / (rad/s)')
    _axes(torque_ax,'（b）双臂实际关节电机驱动力矩','驱动力矩 / (N·m)')
    peak=max(np.nanmax(np.abs(v)) for v in data['velocity'].values())
    velocity_ax.set_ylim(-max(.66,peak*1.10),max(.66,peak*1.10))
    velocity_ax.text(.985,.97,'黑色点线：参考速度上限 ±0.60 rad/s',transform=velocity_ax.transAxes,
                     ha='right',va='top',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
    torque_ax.text(.985,.97,'电机目标力矩上限\n关节 1–3：200 N·m　关节 4–6：50 N·m',
                   transform=torque_ax.transAxes,ha='right',va='top',fontsize=9,
                   bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
    for ax in [velocity_ax,torque_ax]:ax.set_xlim(t[0],t[-1])
    for arm in ['A','B']:
        if np.any(np.isfinite(data['reaction'][arm])):
            force_ax.plot(t,data['reaction'][arm],color=ARM_COLORS[arm],lw=1.6,
                          label=arm+' 臂支撑反力幅值（已去皮）')
    if np.any(np.isfinite(data['load'])):
        force_ax.step(t,data['load'],where='post',color='#7e59a0',ls=(0,(4,2)),lw=1.5,label='实际施加的插装外载幅值')
    else:
        force_ax.text(.02,.05,'该旧日志未记录外加载荷通道',transform=force_ax.transAxes,
                      va='bottom',color='#9c442d',fontsize=9)
    _axes(force_ax,'（c）实际支撑反力与插装外载','力幅值 / N')
    force_ax.set_xlim(*window)
    upper=max(1.,force_ax.get_ylim()[1])
    force_ax.set_ylim(-.3,upper)
    force_ax.legend(loc='upper left',frameon=False,fontsize=8.5)
    _axes(bond_ax,'（d）桥面连接处实际力矩范数','连接力矩范数 / (N·m)')
    if np.any(np.isfinite(data['bond'])):
        bond_ax.plot(t,data['bond'],color='#29586f',lw=1.65,label='beam SnapFT 实测力矩范数')
    else:
        bond_ax.text(.02,.80,'此日志缺少有效 beam SnapFT 通道，无法作力矩验收',transform=bond_ax.transAxes,
                     va='top',color='#9c442d',fontsize=9)
    bond_ax.axhline(SNAP_LIMIT,color='#ae463e',ls=(0,(5,3)),lw=1.3,label='连接力矩限值 1.20 N·m')
    bond_ax.set_xlim(*window);bond_ax.set_ylim(bottom=0.)
    bond_ax.legend(loc='upper left',frameon=False,fontsize=8.5)
    intervals=_intervals(t,np.isfinite(data['load']) & (data['load']>1e-9))
    for ax in axes.flatten():
        for begin,end in intervals:ax.axvspan(begin,end,color='#9ea9b6',alpha=.10,zorder=0)
    fig.text(.075,.052,'力相关子图：'+zoom_note+'；浅灰背景表示日志中实际外加载荷非零。连接尚未建立或传感器无效时不绘制零值替代。',fontsize=9,color='#505b68')
    fig.text(.075,.023,'关节速度与力矩来自仿真器实际读数；支撑反力来自去皮后的 PadFT；连接力矩来自有效 SnapFT 测量。',fontsize=9,color='#505b68')
    name='normal_dynamics_diagnostic' if diagnostic else 'normal_dynamics_validation'
    paths=[]
    for suffix in ['png','svg']:
        path=outdir/(name+'.'+suffix)
        fig.savefig(path,dpi=220,bbox_inches='tight',pad_inches=.20)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution',type=Path,default=ROOT/'results'/'normal'/'execution.json')
    parser.add_argument('--outdir',type=Path,default=ROOT/'results'/'normal')
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    paths=plot_dynamics(args.execution,args.outdir,allow_incomplete=args.allow_incomplete)
    print(json.dumps({'files':paths,'source':str(args.execution)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

