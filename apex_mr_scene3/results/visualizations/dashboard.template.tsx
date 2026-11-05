import { Button, Checkbox, Divider, H1, H2, H3, Row, Select, Stack, Table, Text, useHostTheme, useState } from "cursor/canvas";

const D: any = __DATA__;
const taskNames: Record<string,string> = {supply_K1:"供应 K1",install_K1:"安装 K1",supply_K2:"供应 K2",install_K2:"安装 K2",close_gate:"封闭 Gate"};
const candidateNames: Record<string,string> = {outer_front:"外侧前方",inner_pocket:"内侧口袋",weak_narrow:"窄接触候选"};
const groupNames: Record<string,string> = {support:"临时支撑动作",task:"供应与安装",handoff:"两次缓冲区交接",balance:"峰值负载惩罚"};
const edgeNames: Record<string,string> = {type1:"Type 1 · 同臂顺序／任务",task_type2:"Type 2 · 跨臂任务",motion_type2:"Type 2 · 几何／模式冲突"};
const f=(n:number,d=2)=>Number(n).toFixed(d);

function Caption({children}:any){ return <Text size="small" tone="tertiary">{children}</Text>; }

function ILP(){
  const t=useHostTheme(), I=D.ilp;
  const [inspect,setInspect]=useState("outer_front:before_gate:B");
  const selected=I.variables.find((v:any)=>v.id===inspect);
  const [showConstraints,setShowConstraints]=useState(false);
  const cols=[t.chart.palette.brightOrange,t.chart.palette.blue];
  const matrix=I.tasks.map((task:any)=>[taskNames[task.id],...["A","B"].map(arm=>{
    const v=I.variables.find((x:any)=>x.id===task.id+":"+arm);
    return <Button variant={v.selected?"primary":"secondary"} onClick={()=>setInspect(v.id)}>{v.selected?"● 已选 ":"× 筛选未通过 "}{arm}{v.selected?" · "+f(v.cost):""}</Button>;
  })]);
  return <Stack gap={24}>
    <Stack gap={8}><H2>ILP 选定怎样分工</H2><Text>A 臂起初托举桥梁。临时支撑承载通过后，A 臂释放并供应 K1、K2，B 臂完成安装、验载、回收支撑和封板。</Text></Stack>
    <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(300px,1fr))",gap:32}}>
      <Stack gap={12}><H3>任务 × 机械臂分配</H3><Table headers={["任务","UR5 A","UR5 B"]} rows={matrix}/><Text size="small" tone="secondary">每个任务的另一臂候选均未通过当前端点筛选。此结果包含两次 A → 缓冲区 → B 交接，缓冲区容量为 1。</Text></Stack>
      <Stack gap={12}><H3>代价组成 · J = 108.46 proxy_cost</H3>
        <svg role="img" aria-label="ILP目标函数四项贡献，单位proxy_cost" viewBox="0 0 550 250" style={{width:"100%",color:t.text.secondary}}>
          {[0,10,20,30,40,50,60].map(x=><g key={x}><line x1={142+x*5.4} y1={12} x2={142+x*5.4} y2={190} stroke={t.stroke.tertiary}/><text x={142+x*5.4} y={212} textAnchor="middle" fill="currentColor" fontSize={12}>{x}</text></g>)}
          {I.objective.breakdown.map((b:any,i:number)=><g key={b.group}><text x={132} y={34+i*44} textAnchor="end" fill="currentColor" fontSize={13}>{groupNames[b.group]}</text><rect x={142} y={17+i*44} height={26} width={b.value*5.4} rx={2} fill={i===3?t.text.tertiary:t.accent.primary} opacity={i===3?0.65:1}/><text x={149+b.value*5.4} y={35+i*44} fontSize={12} fill={t.text.primary}>{f(b.value)}</text></g>)}
          <text x={310} y={240} textAnchor="middle" fill="currentColor" fontSize={12}>代价（proxy_cost），纵轴为代价项</text>
        </svg>
        <Caption>来源 allocation.json · 本次已验收运行 · 直接代价 96.76 + 0.15 × 峰值工作量 77.99。此目标使用端点启发式成本。</Caption>
        <Row gap={28} wrap>{["A","B"].map((arm,i)=><div key={arm}><Text weight="semibold" style={{color:cols[i]}}>UR5 {arm}</Text><Text>{f(I.objective.arm_workloads[arm])} proxy_cost</Text></div>)}</Row>
      </Stack>
    </div>
    <Divider/>
    <Stack gap={12}><H3>支撑位置与撤除顺序的六个候选模式</H3><Table headers={["支撑位置","撤除顺序","成本 proxy_cost","最终决策"]} rows={I.variables.filter((v:any)=>v.group==="support").map((v:any)=>[
      candidateNames[v.candidate],v.removal_order==="before_gate"?"封板前撤撑":"封板后撤撑",v.display_cost==null?"未赋有效成本":f(v.display_cost),<Button variant={v.selected?"primary":"secondary"} onClick={()=>setInspect(v.id)}>{v.selected?"● 选中":v.enabled_by_bounds?"可用 · 未选":"筛选未通过"}</Button>
    ])}/><Text size="small" tone="secondary">最终选择外侧前方，由 B 臂放置及回收。原先“封板后撤撑”的完整分配组合在后续 Gate 接近姿态求解中失败，加入组合排除约束后，改为同成本的“封板前撤撑”。</Text></Stack>
    <div style={{padding:18,background:t.fill.quaternary,borderRadius:6}}><Stack gap={8}><H3>选中项检查 · {selected.id}</H3><Text>变量 x = {selected.value}，上界 {selected.upper_bound}，目标贡献 {f(selected.objective_contribution)} proxy_cost。</Text><Text>{selected.selected?"该变量取 1，属于最终解。":selected.enabled_by_bounds?"该变量仍允许取 1。被排除的是此前的完整组合，不能据此判断这个单独模式在所有组合中均不可行。":"该变量上界为 0，当前候选筛选未通过。"}</Text>{selected.rejection_reasons?.length>0&&<Text size="small" tone="secondary">{selected.rejection_reasons.join(" · ")}</Text>}</Stack></div>
    <Stack gap={10}><H3>求解证据与适用范围</H3><Text>HiGHS 返回 Optimal，MIP gap = 0。模型有 18 个二元变量、1 个连续峰值工作量变量和 17 条线性约束，最优范围限于当前候选与已加入的排除约束。</Text><Text size="small" tone="secondary">ILP 未直接优化秒数。图中工作量为代价估计，运动路径与执行时间由后续 RRT-Connect、TPG 和动力学执行确定。</Text><Row><Button onClick={()=>setShowConstraints(!showConstraints)}>{showConstraints?"收起":"展开"} 17 条约束及残差</Button></Row>{showConstraints&&<Table headers={["约束","左值","下界","上界","满足"]} rows={I.constraints.map((c:any)=>[c.constraint,f(c.lhs_at_solution,4),c.lower==null?"—":f(c.lower,4),c.upper==null?"—":f(c.upper,4),c.satisfied?"是":"否"])}/>}<Caption>来源 allocation.json、allocation_problem.json、candidate_qualification.json、alloc_and_reference.json · Run {I.run_id}</Caption></Stack>
  </Stack>;
}

function TPG(){
  const t=useHostTheme(), G=D.tpg;
  const [nodeId,setNodeId]=useState(0), [zoom,setZoom]=useState("1");
  const [type1,setType1]=useState(false),[taskEdges,setTaskEdges]=useState(true),[motionEdges,setMotionEdges]=useState(false);
  const [search,setSearch]=useState("");
  const n=G.nodes[nodeId], W=1100*Number(zoom), max=D.summary.nominal_makespan_s;
  const x=(s:number)=>84+s/max*(W-122), y=(a:string)=>a==="A"?110:280;
  const edgeColor=(type:string)=>type==="type1"?t.text.tertiary:type==="task_type2"?t.accent.primary:t.chart.palette.deepOrange;
  const edges=G.edges.filter((e:any)=>e.source===nodeId||e.target===nodeId||({type1,task_type2:taskEdges,motion_type2:motionEdges} as any)[e.type]);
  const linked=new Set(G.edges.filter((e:any)=>e.source===nodeId||e.target===nodeId).flatMap((e:any)=>[e.source,e.target]));
  const matches=G.nodes.filter((v:any)=>(v.id+v.task_label+v.arm+v.kind).toLowerCase().includes(search.toLowerCase()));
  const detailEdges=G.edges.filter((e:any)=>e.source===nodeId||e.target===nodeId);
  const before=G.modes[n.state],after=G.modes[n.state_end];
  const supportName:Record<string,string>={A_initial:"A 初始托举待验载",A:"A 臂托举",A_fixture:"A 与临时支撑共同承载",fixture:"临时支撑",proof_transfer:"向永久结构转移并验载",permanent:"永久结构"};
  const fixtureName:Record<string,string>={parked:"归仓停放",carried_for_placement:"搬运至安装位",placed:"已放置待验载",loaded:"已承载",retracting:"回缩并卸载",unloaded:"已卸载",carried_for_removal:"夹持回收中",extracted:"已抽出"};
  return <Stack gap={20}>
    <Stack gap={8}><H2>多模态 TPG · 完整 419 节点与 760 条依赖</H2><Text>圆形为 386 个位姿节点，菱形为 33 个技能节点。抓取、放置、承载交接、验载、撤撑等技能会改变持物和接触状态，后继动作由完成事件放行。</Text><Text size="small" tone="secondary">原始 node.mode 字段均为空。这里的操作标签由 payload、skill、effects 与实际事件派生，保留原始节点及全部依赖边。</Text></Stack>
    <Row gap={18} wrap><Checkbox checked={type1} onChange={setType1} label="同臂 Type 1（558）"/><Checkbox checked={taskEdges} onChange={setTaskEdges} label="跨臂任务 Type 2（7）"/><Checkbox checked={motionEdges} onChange={setMotionEdges} label="几何／模式冲突 Type 2（195）"/><label style={{display:"flex",alignItems:"center",gap:8}}>缩放<Select value={zoom} onChange={setZoom} options={[1,2,3,5].map(z=>({value:String(z),label:z+"×"}))}/></label></Row>
    <div style={{overflowX:"auto",border:"1px solid "+t.stroke.secondary,borderRadius:6}}>
      <svg role="img" aria-label="全量多模态TPG依赖图。横轴为名义最早起点秒，纵轴为UR5机械臂。可点击节点" width={W} height={420} viewBox={`0 0 ${W} 420`} style={{display:"block",background:t.bg.editor}}>
        <defs><marker id="tpg-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L8,4 L0,8" fill={t.text.tertiary}/></marker></defs>
        {Array.from({length:9},(_,i)=>i*50).map(s=><g key={s}><line x1={x(s)} y1={36} x2={x(s)} y2={350} stroke={t.stroke.tertiary}/><text x={x(s)} y={376} textAnchor="middle" fill={t.text.secondary} fontSize={12}>{s}</text></g>)}
        {["A","B"].map((a,i)=><g key={a}><rect x={76} y={y(a)-32} width={W-105} height={64} fill={t.fill.quaternary}/><text x={16} y={y(a)+4} fill={i===0?t.chart.palette.brightOrange:t.chart.palette.blue} fontSize={14} fontWeight={600}>UR5 {a}</text></g>)}
        <text x={16} y={28} fill={t.text.secondary} fontSize={12}>机械臂</text>
        {edges.map((e:any,i:number)=>{ const a=G.nodes[e.source],b=G.nodes[e.target],hot=e.source===nodeId||e.target===nodeId, xa=x(a.nominal_start),xb=x(b.nominal_start),ya=y(a.arm),yb=y(b.arm); return <path key={i} d={a.arm===b.arm?`M${xa},${ya} Q${(xa+xb)/2},${ya-(e.reason==="same_arm_order"?12:42)} ${xb},${yb}`:`M${xa},${ya} C${xa},${ya+58} ${xb},${yb-58} ${xb},${yb}`} fill="none" stroke={edgeColor(e.type)} strokeWidth={hot?1.8:0.8} opacity={hot?0.95:e.type==="motion_type2"?0.20:0.45} strokeDasharray={e.type==="motion_type2"?"4 3":undefined} markerEnd="url(#tpg-arrow)"><title>{edgeNames[e.type]} · {a.id} → {b.id} · {e.reason}</title></path>; })}
        {G.nodes.map((v:any)=>{const cx=x(v.nominal_start),cy=y(v.arm),active=v.i===nodeId, r=active?7:v.node_type==="skill"?5:3,fill=active?t.accent.primary:v.arm==="A"?t.chart.palette.brightOrange:t.chart.palette.blue;return <g key={v.i} onClick={()=>setNodeId(v.i)} style={{cursor:"pointer"}} opacity={linked.has(v.i)||active?1:0.72}>
          <circle cx={cx} cy={cy} r={8} fill="transparent"/>{v.node_type==="skill"?<path d={`M${cx},${cy-r} l${r},${r} l-${r},${r} l-${r},-${r} Z`} fill={fill} stroke={t.bg.editor} strokeWidth={0.6}/>:<circle cx={cx} cy={cy} r={r} fill={fill} stroke={t.bg.editor} strokeWidth={0.5}/>}<title>{v.id} · {v.task_label} · {f(v.nominal_start)} s</title>
        </g>;})}
        <text x={W/2} y={410} textAnchor="middle" fontSize={13} fill={t.text.secondary}>TPG 名义最早起点（s） · 横向位置表示时间</text>
      </svg>
    </div>
    <Caption>来源 tpg.json · 名义计划 0–419.23 s · 558 条 Type 1 = 417 条同臂相邻边 + 141 条同臂任务依赖。保留全部 760 条边，其中有 628 个唯一端点对。</Caption>
    <Text size="small" tone="secondary">195 条几何／模式冲突边包含 24 条采样几何冲突、137 条模式不一致和 34 条关节状态模式不一致的保守顺序约束，代表执行先后限制。</Text>
    <Row gap={12} wrap><label style={{display:"flex",gap:8,alignItems:"center"}}>搜索节点<input aria-label="搜索TPG节点" value={search} onChange={e=>setSearch(e.target.value)} placeholder="例如 K1、support、verify" style={{padding:8,color:t.text.primary,background:t.bg.editor,border:"1px solid "+t.stroke.primary,borderRadius:4}}/></label><label style={{display:"flex",gap:8,alignItems:"center"}}>选择节点<Select value={String(nodeId)} onChange={v=>setNodeId(Number(v))} options={(matches.some((v:any)=>v.i===nodeId)?matches:[n,...matches]).map((v:any)=>({value:String(v.i),label:`${v.i+1} · ${v.arm} · ${v.id}`}))} style={{maxWidth:480}}/></label></Row>
    <div style={{padding:18,background:t.fill.quaternary,borderRadius:6}}><Stack gap={10}><H3>{n.task_label} · UR5 {n.arm}</H3><Text size="small">{n.id}</Text><Row gap={28} wrap><Text>{n.node_type==="skill"?"技能节点":"位姿节点"} · {n.kind}</Text><Text>持物 {n.payload||"无"}</Text><Text>技能 {n.skill||"—"}</Text></Row><Text>名义 {f(n.nominal_start)} → {f(n.nominal_end)} s，实际 {f(n.actual_start)} → {f(n.actual_end)} s。</Text><Text size="small" tone="secondary">{n.mode||""}{n.mode_end&&n.mode_end!==n.mode?" → "+n.mode_end:""}</Text></Stack></div>
    {before&&after&&<Stack gap={10}><H3>所选节点前后的操作状态</H3><Table headers={["状态项","节点开始","节点完成"]} rows={[
      ["A 臂持物",before.held.A||"空载",after.held.A||"空载"],
      ["B 臂持物",before.held.B||"空载",after.held.B||"空载"],
      ["承载来源",supportName[before.support]||before.support,supportName[after.support]||after.support],
      ["临时支撑",fixtureName[before.fixture]||before.fixture,fixtureName[after.fixture]||after.fixture],
      ["缓冲区占用",before.buffer||"空",after.buffer||"空"],
      ["已安装积木",before.installed.join("、")||"无",after.installed.join("、")||"无"],
      ["A 臂允许释放",before.A_release_authorized?"是":"否",after.A_release_authorized?"是":"否"],
      ["A 支撑垫启用",before.A_pad_active?"是":"否",after.A_pad_active?"是":"否"]
    ]}/><Caption>状态从实际事件重建。空载仅描述夹持的物体，A 初始托举另由承载来源与支撑垫状态表达。</Caption></Stack>}
    <Stack gap={10}><H3>所选节点的直接依赖 · {detailEdges.length} 条</H3>{detailEdges.length>0&&<Table headers={["方向","依赖类型","相邻节点","原因"]} rows={detailEdges.map((e:any)=>{const v=G.nodes[e.source===nodeId?e.target:e.source];return [e.target===nodeId?"前驱 → 当前":"当前 → 后继",edgeNames[e.type],<Button onClick={()=>setNodeId(v.i)}>{v.id}</Button>,e.reason];})}/>}<Caption>勾选边类型可查看完整依赖。所选节点的邻边始终高亮，点击相邻节点可沿图检查。</Caption></Stack>
  </Stack>;
}

function Execution(){
  const t=useHostTheme(),G=D.tpg;
  const [time,setTime]=useState(116.62),[selected,setSelected]=useState("");
  const W=1100,max=D.summary.actual_makespan_s,x=(s:number)=>85+s/max*970,y=(a:string)=>a==="A"?100:190;
  const active=G.nodes.filter((n:any)=>n.actual_start<=time&&n.actual_end>time);
  const task=G.tasks.find((v:any)=>v.id===selected||v.task_id===selected);
  const ts=(v:any)=>v.actual_start,te=(v:any)=>v.actual_end;
  const milestones=G.events.filter((e:any)=>["support_load_verified","arm_support_released","permanent_structure_verified","support_unloaded","support_extracted","support_returned","gate_closed","assembly_completed"].includes(e.event||e.kind||e.name));
  const eventName:Record<string,string>={support_load_verified:"临时支撑承载通过",arm_support_released:"A 臂释放",permanent_structure_verified:"永久结构验载通过",support_unloaded:"临时支撑卸载",support_extracted:"支撑取出",support_returned:"支撑归仓",gate_closed:"Gate 关闭",assembly_completed:"装配完成"};
  const evtime=(e:any)=>e.time??e.sim_time??e.t;
  const ek=(e:any)=>e.event||e.kind||e.name;
  const eventTime=(key:string)=>evtime(milestones.find((e:any)=>ek(e)===key));
  const holding=G.support_intervals.find((v:any)=>v.id==="initial_A_no_active_nodes");
  const state=time<eventTime("support_load_verified")?"A 臂持续托举桥梁，B 臂布置临时支撑":time<eventTime("arm_support_released")?"临时支撑已验载，等待 A 臂释放":time<eventTime("permanent_structure_verified")?"临时支撑承载，A 供应积木、B 完成装配":time<eventTime("support_unloaded")?"永久结构已验载，准备卸载临时支撑":time<eventTime("support_returned")?"永久结构承载，B 臂回收临时支撑":time<eventTime("gate_closed")?"支撑已归仓，B 臂封板":time<max-0.001?"Gate 已关闭，双臂完成退让":"装配完成，双臂空闲";
  return <Stack gap={20}>
    <Stack gap={8}><H2>真实执行时间 · 双臂异步调度</H2><Text>拖动时间游标查看正在执行的节点与承载状态。甘特条逐段绘制实际节点起止时间，保留等待间隔。点击片段可检查所属任务。</Text></Stack>
    <svg role="img" aria-label="双臂实际参考任务甘特图，横轴仿真秒，纵轴机械臂" viewBox="0 0 1100 300" style={{width:"100%",minHeight:210}}>
      <defs><pattern id="holding-hatch" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0,8 L8,0" stroke={t.chart.palette.brightOrange} strokeWidth={1} opacity={0.6}/></pattern></defs>
      {Array.from({length:10},(_,i)=>i*50).map(s=><g key={s}><line x1={x(s)} y1={34} x2={x(s)} y2={245} stroke={t.stroke.tertiary}/><text x={x(s)} y={270} textAnchor="middle" fontSize={12} fill={t.text.secondary}>{s}</text></g>)}
      {["A","B"].map((a,i)=><g key={a}><text x={16} y={y(a)+6} fontSize={14} fill={t.text.primary}>UR5 {a}</text><line x1={80} y1={y(a)} x2={1060} y2={y(a)} stroke={t.stroke.primary}/></g>)}
      <text x={16} y={22} fontSize={12} fill={t.text.secondary}>机械臂</text>
      <rect x={x(holding.actual_start)} y={y("A")-17} width={x(holding.actual_end)-x(holding.actual_start)} height={34} fill="url(#holding-hatch)" stroke={t.chart.palette.brightOrange}/><text x={x(holding.actual_end/2)} y={y("A")-25} textAnchor="middle" fontSize={12} fill={t.text.secondary}>A 持续托举</text>
      {G.nodes.map((v:any,i:number)=><g key={i} onClick={()=>setSelected(v.task_id)} style={{cursor:"pointer"}}><rect x={x(ts(v))} y={y(v.arm)-17} width={Math.max(0.8,x(te(v))-x(ts(v)))} height={34} rx={1} fill={v.arm==="A"?t.chart.palette.brightOrange:t.chart.palette.blue} stroke={v.task_id===selected?t.text.primary:t.bg.editor} strokeWidth={v.task_id===selected?1.2:0.4}/><title>{v.task_label} · {v.id} · {f(ts(v))}–{f(te(v))} s</title></g>)}
      <line x1={x(time)} y1={26} x2={x(time)} y2={240} stroke={t.text.primary} strokeWidth={1.5} strokeDasharray="4 3"/><text x={Math.min(1000,Math.max(120,x(time)))} y={19} fontSize={13} fill={t.text.primary} textAnchor="middle">{f(time)} s</text>
      <text x={550} y={298} textAnchor="middle" fontSize={13} fill={t.text.secondary}>实际仿真时间（s） · 橙色 UR5 A，蓝色 UR5 B</text>
    </svg>
    <Caption>来源 execution.json · 0–455.60 s · 全部 419 个节点，属于 134 个参考任务。A 释放前的持续托举以斜线表示，不能由主动运动节点的空白推断为空闲。</Caption>
    <label style={{display:"flex",alignItems:"center",gap:16}}>实际时间 {f(time)} s<input aria-label="实际仿真时间游标" type="range" min={0} max={max} step={0.02} value={time} onChange={e=>setTime(Number(e.target.value))} style={{flex:1,accentColor:t.accent.primary}}/></label>
    <div style={{padding:18,background:t.fill.quaternary,borderRadius:6}}><Stack gap={10}><H3>{state}</H3>{active.length>0?<Table headers={["机械臂","执行节点","操作"]} rows={active.map((n:any)=>[n.arm,n.id,n.task_label])}/>:<Text>此时没有主动执行节点。</Text>}{task&&<Text size="small" tone="secondary">所选任务 {task.label||task.task_label||task.id}，{f(ts(task))}–{f(te(task))} s。</Text>}</Stack></div>
    <Stack gap={12}><H3>承载与支撑生命周期</H3><Table headers={["实际时间 s","完成事件","检查"]} rows={milestones.map((e:any)=>[<Button onClick={()=>setTime(Math.min(max,evtime(e)+0.02))}>{f(evtime(e))} s</Button>,eventName[ek(e)],ek(e)])}/><Text size="small" tone="secondary">临时支撑交接测得 1.9605 N。永久结构验载时，每个成功物理回调施加 8 N，连续合格记录 0.70 s，永久连接测得 13.64–14.06 N，临时支撑残余反力最高 0.00119 N。</Text></Stack>
    <Stack gap={10}><H3>时间读数</H3><Table headers={["读数","数值","含义"]} rows={[["ILP 目标值","108.46 proxy_cost","端点启发式成本 + 峰值工作量惩罚"],["TPG 名义完工",f(D.summary.nominal_makespan_s)+" s","按节点名义时长与依赖计算"],["实际动力学完工",f(D.summary.actual_makespan_s)+" s","事件日志测得，含控制与技能耗时"]]}/><Text size="small" tone="secondary">本次模拟使用理想刚性抓取与连接约束。以上结果来自一次完整验收运行。</Text><Caption>来源 summary.json、execution.json、independent_final_audit.json · Run {D.summary.run_id}</Caption></Stack>
  </Stack>;
}

export default function Scene3Results(){
  const t=useHostTheme();
  const [tab,setTab]=useState("ilp");
  return <Stack gap={24} style={{padding:28,maxWidth:1260,margin:"0 auto",fontFamily:"inherit",background:t.bg.editor,color:t.text.primary}}>
    <Stack gap={8}><Text size="small" tone="secondary">APEX-MR / 双 UR5 / SCENE 03</Text><H1>临时支撑与机械臂释放 · 求解结果</H1><Text tone="secondary">同一验收运行的 ILP 决策、多模态时间依赖图和实际执行记录。</Text><Row gap={24} wrap><Text weight="semibold">ILP {f(D.ilp.objective.value)} proxy_cost</Text><Text weight="semibold">TPG 419 节点 / 760 边</Text><Text weight="semibold">实际完成 {f(D.summary.actual_makespan_s)} s</Text></Row></Stack>
    <Row gap={8} wrap>{[["ilp","01 · ILP 决策"],["tpg","02 · 多模态 TPG"],["execution","03 · 实际时序"]].map(([key,label])=><Button key={key} variant={tab===key?"primary":"secondary"} style={{minHeight:36,padding:"8px 16px"}} onClick={()=>setTab(key)}>{label}</Button>)}</Row>
    <Divider/>
    {tab==="ilp"?<ILP/>:tab==="tpg"?<TPG/>:<Execution/>}
    <Divider/><Caption>运行标识 {D.summary.run_id} · 数据直接提取自 D:/APEX_MR_Plus/apex_mr_scene3/results/apex/ · 可视化未修改规划、场景或验收文件。</Caption>
  </Stack>;
}
