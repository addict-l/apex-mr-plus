-- Appended to the existing normal/controller.lua when building a demonstration.
-- The original servo, grasp, snap, force sensing and dependency dispatcher
-- remain in use for both phases. Only this scene enables the fault branch.
local demo, phase, faultAt, macro, recoveryEnabled, liveGate, hostAccepted
local ui, demoUI, interruptedTask, macroStarts, prefixCompleted, contactSamples
local baselineInit = sysCall_init
local baselineActuation = sysCall_actuation
local baselineSensing = sysCall_sensing
local baselineCleanup = sysCall_cleanup
local baselineBegin = beginNode
local baselineFinish = finishNode
local descriptions = {
    place='B 将连接砖放到共享缓存位', acquire_support='B 接替支撑，A 继续托住桥面',
    pick='A 从缓存位取回连接砖', install='A 安装连接砖，B 保持支撑',
}

local function macroText(action)
    if not action then return '' end
    if action.kind=='release_support' then return action.arm..' 解除支撑并撤回' end
    return descriptions[action.kind] or action.kind
end

local function display()
    if not ui then return end
    local text = {normal='正常装配 · 正在运行原任务图', fault='故障 · B 的连接砖安装动作被阻止',
        recovery='恢复 · '..(macro and macroText(macro) or '正在接入剩余计划'), done='恢复完成 · 连接砖已安装'}
    demoUI.setLabelText(ui,1,failure and ('停止 · '..failure) or text[phase] or phase)
    local a = support.A and support.A.ready and '支撑桥面' or held.A and ('持有 '..held.A) or '空手'
    local b = support.B and support.B.ready and '支撑桥面' or held.B and ('持有 '..held.B) or '空手'
    demoUI.setLabelText(ui,2,'A  '..a..'     B  '..b)
    demoUI.setLabelText(ui,3,recoveryEnabled and '故障后自动切换到恢复计划' or '故障处保持等待，可再次点击以恢复')
end

function toggleDemoRecovery()
    if phase=='normal' or phase=='fault' then recoveryEnabled=not recoveryEnabled; display() end
end

function sysCall_init()
    baselineInit()
    demo=assert(sim.readCustomTableData(cfg.root,'RecoveryDemo'))
    phase='normal';faultAt=nil;macro=nil;interruptedTask=nil
    recoveryEnabled=true;liveGate=false;hostAccepted=false;prefixCompleted=0
    macroStarts={};contactSamples={}
    if not sim.getBoolParam(sim.boolparam_headless) then
        demoUI=require 'simUI'
        ui=demoUI.create([[<ui title="双臂装配 · 异常恢复演示" closeable="false" resizable="false" placement="relative" position="-20,40" layout="vbox">
            <label id="1" text="正常装配" style="font-size: 15px; font-weight: bold; min-width: 325px;"/>
            <label id="2" text=""/><label id="3" text=""/>
            <button text="切换：自动恢复 / 保持等待" on-click="toggleDemoRecovery"/>
        </ui>]])
    end
    display()
end

local function snapshot()
    local connections, sustaining, positions = {}, {}, {}
    for name,h in pairs(snapSensors) do
        if sim.isDynamicallyEnabled(h) then connections[#connections+1]=name end
    end
    table.sort(connections)
    for arm,s in pairs(support) do
        if s.ready then sustaining[arm]={part=s.part,force_N=supportForce(arm)} end
    end
    for name,h in pairs(blocks) do positions[name]=sim.getObjectPose(h,sim.handle_world) end
    return {held={A=held.A,B=held.B},support=sustaining,connections=connections,
            q={A=qactual('A'),B=qactual('B')},part_poses=positions}
end

local function compatible()
    if held.B~='cap' or held.A or active.A or active.B or not support.A or not support.A.ready or support.B then
        return false,'holding or support state differs from recovery input'
    end
    if supportForce('A')<=.15 then return false,'initial supporting contact has insufficient measured reaction' end
    for _,part in ipairs(cfg.parts) do
        if part.id~='cap' then
            if not snapSensors[part.id] or not sim.isDynamicallyEnabled(snapSensors[part.id]) then
                return false,'missing established connection '..part.id
            end
            if dist(sim.getObjectPosition(blocks[part.id],sim.handle_world),part.goal)>.009 then
                return false,'installed part outside recovery envelope: '..part.id
            end
        end
    end
    for _,arm in ipairs({'A','B'}) do
        local measured=qactual(arm)
        for i,v in ipairs(demo.handoff_q[arm]) do
            if math.abs(delta(measured[i],v))>.035 then return false,'unexpected joint pose for '..arm end
        end
    end
    return true
end

beginNode = function(node,t)
    if phase=='normal' and node.task_id==demo.fault_task then
        phase='fault';faultAt=t;interruptedTask=node.task_id
        for _,s in pairs(states) do if s=='done' then prefixCompleted=prefixCompleted+1 end end
        log('injected_installation_block',node,{snapshot=snapshot()})
        display()
        return
    end
    if phase=='fault' then return end
    if phase=='recovery' then
        local id=node.metadata.macro_id
        if id and (not macro or id~=macro.id) then
            for _,a in ipairs(demo.recovery_actions) do if a.id==id then macro=a;break end end
            assert(macro and macro.id==id,'unknown recovery action')
            macroStarts[#macroStarts+1]=id;log('recovery_action_started',node,{action=macro.kind,arm=macro.arm,id=id})
        end
        if node.kind=='support_start' and node.arm=='B' then
            node.metadata.support_height_bias_m=demo.replacement_contact_bias_m
        elseif node.kind=='support_end' and node.arm=='A' then
            if not support.B or not support.B.ready or supportForce('B')<=.15 then
                fail('Cannot withdraw A: B support not confirmed');return
            end
            log('support_transfer_confirmed',node,{replacement_force_N=supportForce('B'),original_force_N=supportForce('A')})
        elseif node.kind=='support_end' and node.arm=='B' and not snapSensors.cap then
            fail('Cannot withdraw B before cap is connected');return
        end
    end
    baselineBegin(node,t)
end

local function placeOnBuffer(node)
    local arm,part=node.arm,node.part
    if held[arm]~=part or not graspSensors[part] then fail('Buffer placement ownership mismatch');return end
    local pose=sim.getObjectPose(blocks[part],sim.handle_world)
    if dist(pose,node.metadata.cache_pose)>.009 then fail('Buffer alignment outside 9 mm');return end
    sim.setObjectParent(blocks[part],cfg.root,true)
    sim.removeObjects({graspSensors[part]});graspSensors[part]=nil
    sim.resetDynamicObject(blocks[part]);held[arm]=nil
    log('buffer_placement',node,{measured_pose=pose})
end

finishNode = function(info,t)
    if info.node.kind=='cache' then
        placeOnBuffer(info.node)
        if failure then return end
    end
    baselineFinish(info,t)
    if phase=='recovery' then display() end
end

local function activateRecovery()
    local ok,why=compatible()
    if not ok then fail(why);return end
    program=demo.recovery_program;nodes={};states={};pred={};active={}
    for _,n in ipairs(program.nodes) do nodes[n.id]=n;states[n.id]='pending';pred[n.id]={} end
    for _,e in ipairs(program.edges) do table.insert(pred[e.target],e.source) end
    phase='recovery';log('recovery_plan_installed',nil,{from_live_python=hostAccepted,snapshot=snapshot()});display()
end

function sysCall_actuation()
    if phase=='fault' and not failure then
        pressLoad=nil
        for side,a in pairs(arms) do target(side,a.hold) end
        if sim.getSimulationTime()-faultAt>=demo.fault_pause_s and recoveryEnabled and (not liveGate or hostAccepted) then
            activateRecovery()
        end
        return
    end
    baselineActuation()
    if done and phase=='recovery' then phase='done';display() end
    if failure then display() end
end

function sysCall_sensing()
    baselineSensing()
    if phase=='recovery' and pressLoad and stepCount%5==0 then
        contactSamples[#contactSamples+1]={time=sim.getSimulationTime(),load_N=pressLoad.force,
            support_B_N=supportForce('B'),support_B_ready=support.B and support.B.ready or false}
    end
end

function requireLiveRecovery(value)
    if phase=='normal' or phase=='fault' then liveGate=value;return true end
    return false
end

function acceptLiveRecovery(rows)
    if phase~='fault' or #rows~=#demo.recovery_actions then return false end
    for i,a in ipairs(rows) do
        local stored=demo.recovery_actions[i]
        if a.kind~=stored.kind or a.arm~=stored.arm or a.target~=stored.target
                or a.destination~=stored.destination or a.receiver~=stored.receiver then return false end
    end
    hostAccepted=true;return true
end

function getDemoState()
    local state=getState()
    state.phase=phase;state.fault_time=faultAt;state.fault_injected=faultAt~=nil
    state.interrupted_task=interruptedTask;state.prefix_completed=prefixCompleted
    state.live_host_accepted=hostAccepted;state.macro=macro and macro.id or nil
    state.recovery_enabled=recoveryEnabled
    state.panel_visible=ui~=nil
    return state
end

function getDemoSnapshot() return snapshot() end
function getDemoResult()
    local result=getDemoState();local physical=snapshot()
    result.part_poses=physical.part_poses;result.held=physical.held;result.support=physical.support
    result.connections=physical.connections;result.macro_starts=macroStarts;result.events=events
    result.insertion_samples=contactSamples;result.dynamic_enablement=enabled
    result.contact_trace_tail={}
    for i=math.max(1,#trace-39),#trace do
        local row=trace[i];local moments={}
        for part,bond in pairs(row.snap) do moments[part]=bond.torque_norm_Nm end
        table.insert(result.contact_trace_tail,{time=row.time,support=row.support,moments=moments,load_N=row.applied_insertion_load_N})
    end
    return result
end

function sysCall_cleanup()
    if ui then demoUI.destroy(ui);ui=nil end
    baselineCleanup()
end
