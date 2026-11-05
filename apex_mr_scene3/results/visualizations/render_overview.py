"""Shareable Chinese figure derived only from the accepted scene-3 run.

Reads accepted artifacts; never changes a planner, controller, or input result.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SRC = HERE.parent/'apex'
RUN = '20261103T121253Z-d0d0c68e'
BG, INK, MUTED, LINE = '#f3f6f9', '#193245', '#607587', '#dce5eb'
A, B, PURPLE = '#b76b20', '#117d87', '#7d609e'


def read(name):
    return json.loads((SRC/name).read_text(encoding='utf-8'))


def load_data():
    summary, graph, execution, allocation, audit = [read(name) for name in
        ['summary.json', 'tpg.json', 'execution.json', 'allocation.json', 'execution_validation.json']]
    assert all(row['run_id'] == RUN for row in (summary, execution, audit))
    assert summary['success'] and summary['main_scene_promoted'] and summary['stage'] == 'actual_execution_audited'
    assert execution['success'] and audit['success'] and not audit['violations']
    counts = Counter(node['node_type'] for node in graph['nodes'])
    edges = Counter('type1' if edge['type'] == 'type1' else edge['reason'] for edge in graph['edges'])
    assert counts == {'pose': 386, 'skill': 33}
    assert edges == {'type1': 558, 'task_dependency': 7, 'motion_conflict': 195}
    motion_reasons = Counter(edge['evidence']['problem']['reason'] for edge in graph['edges']
                             if edge.get('reason') == 'motion_conflict')
    assert motion_reasons == {'interarm_collision':18, 'payload_other_arm_collision:K1:A':3,
                              'payload_payload_collision':3, 'inconsistent_mode':137,
                              'inconsistent_articulation_mode':34}
    assert allocation['cost_unit'] == 'proxy_cost' and allocation['objective_is_makespan'] is False
    assert allocation['task_assignments'] == {'supply_K1':'A','supply_K2':'A','install_K1':'B','install_K2':'B','close_gate':'B'}
    support = allocation['selected_support']
    assert (support['candidate'], support['removal_order'], support['placer_arm'], support['remover_arm']) == ('outer_front','before_gate','B','B')
    times, events = defaultdict(dict), {}
    for event in execution['events']:
        kind = event['kind']
        if kind in ('node_started', 'node_completed'):
            field = 'start' if kind == 'node_started' else 'end'
            assert field not in times[event['node']]
            times[event['node']][field] = event['time']
        elif kind in ('support_load_verified','arm_support_released','permanent_structure_verified',
                       'support_unloaded','support_extracted','support_returned','gate_closed'):
            assert kind not in events
            events[kind] = event
    rows = []
    for node in graph['nodes']:
        interval = times[node['id']]
        assert set(interval) == {'start','end'} and interval['end'] >= interval['start']
        rows.append({**node, **interval})
    assert math.isclose(max(row['end'] for row in rows), summary['actual_makespan_s'], abs_tol=.03)
    # Four displayed cross-arm semantic relations are exactly the seven task
    # Type-2 edges, with the two endpoint targets of each handoff aggregated.
    task2 = [edge for edge in graph['edges'] if edge['type']=='type2' and edge['reason']=='task_dependency']
    bundles = Counter()
    for edge in task2:
        source, target = edge['source_task'], edge['target_task']
        if source == 'verify_support' and target == 'release_A': key = 'release'
        elif source == 'deposit_K1_retreat_contact_clear' and target in ('pick_K1_approach','pick_K1'): key = 'K1'
        elif source == 'pick_K1' and target in ('deposit_K2_approach','deposit_K2'): key = 'capacity'
        elif source == 'deposit_K2_retreat_contact_clear' and target in ('pick_K2_approach','pick_K2'): key = 'K2'
        else: raise AssertionError('Unknown task dependency '+str(edge))
        bundles[key] += 1
    assert bundles == {'release':1, 'K1':2, 'capacity':2, 'K2':2}
    return summary, graph, execution, allocation, audit, rows, events, counts, edges, bundles


def text(ax, x, y, value, size=11, color=INK, **kwargs):
    return ax.text(x,y,value,fontsize=size,color=color,va='center',**kwargs)


def box(ax, x, y, w, h, color='white', edge=LINE, radius=.018, **kwargs):
    patch = FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=color,edgecolor=edge,linewidth=.8,**kwargs)
    ax.add_patch(patch)
    return patch


def panel(fig, rect):
    ax=fig.add_axes(rect);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    box(ax,0,0,1,1)
    return ax


def heading(ax, index, title, subtitle):
    text(ax,.035,.925,index,12,B,weight='bold')
    text(ax,.092,.925,title,16,weight='bold')
    text(ax,.035,.84,subtitle,9.4,MUTED)


def category(task, side):
    if side == 'A':
        if task == 'A_final_home': return '归位', '#7b8999'
        if 'K1' in task: return 'K1 供料', '#c17c31'
        if 'K2' in task: return 'K2 供料', '#95672c'
        return '释放 / 撤撑', '#dfb67c'
    if task == 'B_final_home': return '归位', '#7b8999'
    if 'gate' in task: return '挡板装配', '#7d609e'
    if 'K1' in task: return 'K1 取料 / 装配', '#117d87'
    if 'K2' in task: return 'K2 取料 / 装配', '#3974a2'
    if task in ('verify_permanent','unload_support'): return '承载验证 / 卸载', '#49815c'
    if any(word in task for word in ('remove_support','regrasp_support','raise_unloaded','extract','return_support')):
        return '拆撑 / 归仓', '#468c86'
    if task.startswith('B_clear_fixture'): return '夹具撤离', '#74aeb3'
    return '工具搬运 / 支撑接替', '#9dc9c7'


def render():
    summary, graph, execution, allocation, audit, rows, events, counts, edges, bundles = load_data()
    font = Path('C:/Windows/Fonts/msyh.ttc')
    assert font.exists(), 'Chinese font is required; never export missing glyphs.'
    font_manager.fontManager.addfont(str(font))
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyhbd.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path'})
    fig=plt.figure(figsize=(18,13.4),facecolor=BG)
    fig.text(.04,.963,'APEX-MR  /  双 UR5 积木装配',fontsize=24,weight='bold',color=INK,va='top')
    fig.text(.04,.921,'ILP 任务分配  →  多模态 TPG  →  实际事件执行',fontsize=13,color=MUTED)
    fig.text(.96,.955,'SCENE 03',fontsize=17,weight='bold',ha='right',color=B)
    fig.text(.96,.924,'单次完整动力学验收通过',fontsize=11,ha='right',color=B)

    left=panel(fig,[.04,.594,.443,.296])
    heading(left,'01','ILP 选中了什么','HiGHS / MILP  ·  当前有限候选与几何反馈约束下最优')
    box(left,.035,.465,.44,.295,'#fcf4e8','#eedcc4')
    box(left,.525,.465,.44,.295,'#e9f4f4','#cce4e5')
    text(left,.065,.705,'A / 左臂',13,A,weight='bold')
    text(left,.065,.620,'供应 K1、K2',16,A,weight='bold')
    text(left,.065,.525,'2 次交接 → 单格共享缓存',10,MUTED)
    text(left,.555,.705,'B / 右臂',13,B,weight='bold')
    text(left,.555,.620,'安装 K1、K2、Gate',15,B,weight='bold')
    text(left,.555,.525,'放置临时支撑 + 拆撑归仓',10,MUTED)
    box(left,.035,.295,.93,.125,'#f4f7fa')
    text(left,.06,.368,'outer_front',12,weight='bold')
    text(left,.365,.368,'before_gate  /  先拆撑，再关挡板',11)
    text(left,.06,.322,'选中支撑模式',9,MUTED)
    text(left,.365,.322,'支撑放置与回收：B',9,MUTED)
    text(left,.035,.209,f"J = {allocation['objective']:.7f}",22,weight='bold')
    text(left,.035,.112,'单位：proxy_cost（代理代价，不是秒）',11,A,weight='bold')
    text(left,.035,.051,'J = 所选模式代价之和 + 0.15 × 双臂峰值负载代理',9.3,MUTED)

    right=panel(fig,[.503,.594,.457,.296])
    heading(right,'02','多模态 TPG 的真实构成','419 个节点 / 760 条边  ·  路径节点与技能节点共同执行')
    # Count bars are proportional to the actual validated node/edge counts.
    text(right,.035,.746,'节点',10,MUTED)
    for start,count,color in [(0,counts['pose'],'#487d9d'),(counts['pose'],counts['skill'],'#d99842')]:
        box(right,.13+.82*start/419,.705,.82*count/419,.075,color,color,radius=0)
    text(right,.13,.650,f"pose 位姿 {counts['pose']}",10,'#487d9d',weight='bold')
    text(right,.95,.650,f"skill 技能 {counts['skill']}",10,'#b67523',ha='right',weight='bold')
    text(right,.035,.554,'依赖',10,MUTED)
    offset=0
    for count,color in [(edges['type1'],'#667f92'),(edges['task_dependency'],B),(edges['motion_conflict'],PURPLE)]:
        box(right,.13+.82*offset/760,.515,.82*count/760,.075,color,color,radius=0);offset+=count
    text(right,.13,.462,'Type-1  558',10,'#667f92',weight='bold')
    text(right,.13,.411,'同臂节点与任务顺序',9,MUTED)
    text(right,.44,.462,'任务 Type-2  7',10,B,weight='bold')
    text(right,.44,.411,'跨臂语义依赖',9,MUTED)
    text(right,.735,.462,'避碰 Type-2  195',10,PURPLE,weight='bold')
    text(right,.735,.411,'几何 / 模式冲突',9,MUTED)
    text(right,.13,.350,'195 条 = 24 条采样几何冲突 + 171 条模式保守约束',9,MUTED)
    text(right,.035,.295,'pose → 沿 RRT 路径运动',11,'#487d9d',weight='bold')
    text(right,.035,.219,'skill → 抓取、连接、承载检查、卸载与归仓',11,'#b67523',weight='bold')
    text(right,.035,.120,'节点完成后才释放后继；技能守卫须有实测证据。',10,MUTED)
    text(right,.035,.055,'下图只聚合 7 条任务 Type-2 边；完整节点与避碰边未逐一展开。',9,MUTED)

    # Exact semantic grouping of all seven cross-arm task-dependency edges.
    semantic=panel(fig,[.04,.389,.92,.184])
    text(semantic,.018,.865,'03',12,B,weight='bold')
    text(semantic,.047,.865,'跨臂任务依赖：7 条边聚合为 4 组交接',14,weight='bold')
    text(semantic,.98,.865,'语义聚合视图  ·  非完整 419 节点图',10,MUTED,ha='right')
    text(semantic,.026,.60,'A',14,A,weight='bold')
    text(semantic,.026,.235,'B',14,B,weight='bold')
    pos={'release':(.17,.60), 'dep1':(.415,.60), 'dep2':(.735,.60),
         'verify':(.105,.235),'pick1':(.555,.235),'pick2':(.89,.235)}
    labels={'release':'释放 A','dep1':'K1 交付并清场','dep2':'K2 交付并清场',
            'verify':'支撑承载验证','pick1':'取 K1','pick2':'取 K2'}
    for name,(x,y) in pos.items():
        arm_a=name in ('release','dep1','dep2')
        box(semantic,x-.070,y-.095,.14,.19,'#fcf4e8' if arm_a else '#e9f4f4',radius=.012)
        text(semantic,x,y,labels[name],11,A if arm_a else B,ha='center',weight='bold')
    for src,dst,key,note in [('verify','release','release','1  承载接替'),('dep1','pick1','K1','2  物料交接'),
                            ('pick1','dep2','capacity','2  单格缓存释放'),('dep2','pick2','K2','2  物料交接')]:
        x1,y1=pos[src];x2,y2=pos[dst]
        start=(x1,y1+(.10 if y2>y1 else -.10));end=(x2,y2+(-.10 if y2>y1 else .10))
        semantic.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,linewidth=1.5,color=B))
        text(semantic,(x1+x2)/2+.01,(y1+y2)/2,note,9,B,ha='left',bbox={'facecolor':'white','edgecolor':'none','pad':1.7})

    # Actual intervals: preserve the 419 event windows, including wait gaps.
    timeline_panel=panel(fig,[.04,.116,.92,.253])
    text(timeline_panel,.018,.91,'04',12,B,weight='bold')
    text(timeline_panel,.047,.91,'双臂实际执行甘特',14,weight='bold')
    text(timeline_panel,.98,.91,f"完工 {summary['actual_makespan_s']:.2f} s",15,INK,ha='right',weight='bold')
    text(timeline_panel,.018,.807,'按语义分组着色；每一条段来自实际节点事件，等待间隙保留。初始扶持单独以释放事件界定。',9.4,MUTED)
    ax=fig.add_axes([.092,.165,.844,.141])
    ax.set_facecolor('white');ax.set_xlim(0,465);ax.set_ylim(-.48,1.55)
    ax.set_yticks([1,0],['A / 左臂','B / 右臂']);ax.tick_params(axis='both',labelsize=10,length=0,pad=7)
    ax.set_xticks(list(range(0,451,50)));ax.set_xlabel('实际仿真时间 / s',fontsize=10,labelpad=6,color=MUTED)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.grid(axis='x',color=LINE,linewidth=.7);ax.set_axisbelow(True)
    released=events['arm_support_released']['time']
    ax.barh(1,released,left=0,height=.34,color='#fff3dd',edgecolor='#d0aa71',hatch='///',linewidth=.5)
    text(ax,released/2,1,'初始持续扶持',10,A,ha='center')
    groups=defaultdict(list)
    for row in rows:
        label,color=category(row['task_id'],row['arm']);y=1 if row['arm']=='A' else 0
        ax.barh(y,row['end']-row['start'],left=row['start'],height=.34,color=color,
                edgecolor='white' if row['node_type']=='pose' else INK,
                linewidth=.13 if row['node_type']=='pose' else .45,zorder=3)
        groups[(row['arm'],label,color)].append(row)
    # One concise label per semantic group; tiny confirmation phases are named
    # in the milestone cards, avoiding illegible labels on narrow intervals.
    for (arm,label,color),members in groups.items():
        begin=min(r['start'] for r in members);end=max(r['end'] for r in members)
        if label == '承载验证 / 卸载':continue
        short={'K1 取料 / 装配':'K1 装配','K2 取料 / 装配':'K2 装配','工具搬运 / 支撑接替':'搬运工具 / 接替','拆撑 / 归仓':'拆撑 / 归仓'}.get(label,label)
        text(ax,(begin+end)/2,(1 if arm=='A' else 0)+.295,short,9.2,color,ha='center')
    key_events=['support_load_verified','permanent_structure_verified','support_returned','gate_closed']
    for i,key in enumerate(key_events,1):
        time=events[key]['time']
        ax.axvline(time,color=B,linestyle=(0,(2,3)),linewidth=.8,alpha=.65,zorder=2)
        ax.annotate(str(i),(time,-.35),ha='center',va='center',fontsize=8,color='white',
                    bbox={'boxstyle':'circle,pad=.18','facecolor':B,'edgecolor':'white','linewidth':.6})
    # Lifecycle cards tie the Gantt markers directly to actual semantic events.
    card_info=[('支撑接替',f"{events['support_load_verified']['time']:.2f} s · {events['support_load_verified']['details']['force_N']:.3f} N",f"A 释放：{released:.2f} s"),
               ('永久承载确认',f"{events['permanent_structure_verified']['time']:.2f} s · 实际 8 N 验证",f"合格连续区间：{audit['permanent_load_test_observed_duration_s']:.2f} s"),
               ('支撑工具归仓',f"{events['support_returned']['time']:.2f} s",'回缩卸载 → 抽出 → 运回工具位'),
               ('挡板闭合',f"{events['gate_closed']['time']:.2f} s",f"双臂最终空手；完工 {summary['actual_makespan_s']:.2f} s")]
    for i,(title,value,note) in enumerate(card_info):
        c=panel(fig,[.04+i*.2335,.031,.2195,.072])
        text(c,.055,.73,str(i+1),11,B,weight='bold');text(c,.15,.73,title,11,weight='bold')
        text(c,.055,.43,value,10,B,weight='bold');text(c,.055,.14,note,8.1,MUTED)
    fig.text(.04,.012,f'来源运行：{RUN}  |  allocation.json · tpg.json · execution.json · execution_validation.json',fontsize=8.5,color=MUTED)
    fig.text(.96,.012,'理想刚性抓取 / 连接的单次仿真；不报告实测加速比。',fontsize=8.5,ha='right',color=MUTED)
    HERE.mkdir(exist_ok=True)
    outputs=[]
    for suffix in ('png','svg'):
        path=HERE/f'scene3_ilp_tpg_overview.{suffix}'
        fig.savefig(path,dpi=200,facecolor=BG)
        outputs.append(str(path))
    plt.close(fig)
    provenance={'run_id':RUN,'inputs':{name:hashlib.sha256((SRC/name).read_bytes()).hexdigest() for name in
                ['summary.json','allocation.json','tpg.json','execution.json','execution_validation.json']},
                'scope':'ILP assignments, exact TPG counts, four bundles of seven task Type-2 edges, actual node-event Gantt; no nominal/actual speedup claim',
                'source_node_count':len(rows),'task_type2_bundles':dict(bundles),'outputs':outputs,'chinese_font':str(font)}
    provenance_path = HERE/'overview_provenance.json'
    provenance_text = json.dumps(provenance,ensure_ascii=False,indent=2)
    if not provenance_path.exists() or provenance_path.read_text(encoding='utf-8') != provenance_text:
        provenance_path.write_text(provenance_text,encoding='utf-8')
    print(json.dumps(provenance,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    render()
