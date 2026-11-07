-- Appended to the existing normal/controller.lua when building a demonstration.
-- The original servo, grasp, snap, force sensing and dependency dispatcher
-- remain in use for both phases. Only this scene enables the fault branch.
local demo, phase, faultAt, macro, recoveryEnabled, liveGate, hostAccepted
local ui, demoUI, interruptedTask, macroStarts, prefixCompleted, contactSamples
local retry, firstFaultAt, recoveryUsed
local beginAttempt, snapshot
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
    local text = {normal=retry and retry.succeeded and '重试成功 · 继续原任务图' or '正常装配 · 正在运行原任务图',
        retry='安装尝试 · '..(retry and retry.attempt or 0)..' / '..demo.retry_config.max_attempts,
        retreat='本次未完成 · 撤回原接近位后重试',fault='重试耗尽 · 准备重新分配任务',
        recovery='恢复 · '..(macro and macroText(macro) or '正在接入剩余计划'), done='装配完成 · 连接砖已安装'}
    demoUI.setLabelText(ui,1,failure and ('停止 · '..failure) or text[phase] or phase)
    local a = support.A and support.A.ready and '支撑桥面' or held.A and ('持有 '..held.A) or '空手'
    local b = support.B and support.B.ready and '支撑桥面' or held.B and ('持有 '..held.B) or '空手'
    demoUI.setLabelText(ui,2,'A  '..a..'     B  '..b)
    demoUI.setLabelText(ui,3,'模式：'..(demo.retry_config.failure_mode=='persistent' and '持续失败' or '暂时失败，最后一次成功')..
        '；共 '..demo.retry_config.max_attempts..' 次尝试'..(recoveryEnabled and '' or '；耗尽后保持等待'))
end

function setDemoRetryConfig(maxAttempts,mode)
    if phase~='normal' or (retry and retry.attempt>0) then return false end
    if type(maxAttempts)~='number' or maxAttempts%1~=0 or maxAttempts<1 then return false end
    if mode~='persistent' and mode~='transient' then return false end
    demo.retry_config.max_attempts=maxAttempts;demo.retry_config.failure_mode=mode
    display();return true
end

function toggleDemoFaultMode()
    return setDemoRetryConfig(demo.retry_config.max_attempts,
        demo.retry_config.failure_mode=='persistent' and 'transient' or 'persistent')
end

function toggleDemoRecovery()
    if phase=='normal' or phase=='fault' then recoveryEnabled=not recoveryEnabled; display() end
end

function sysCall_init()
    baselineInit()
    demo=assert(sim.readCustomTableData(cfg.root,'RecoveryDemo'))
    phase='normal';faultAt=nil;macro=nil;interruptedTask=nil
    firstFaultAt=nil;recoveryUsed=false;retry={attempt=0,succeeded=false,failed_attempts=0}
    assert(demo.retry_config and demo.retry_skill,'Rebuild this scene to enable bounded skill retries')
    recoveryEnabled=true;liveGate=false;hostAccepted=false;prefixCompleted=0
    macroStarts={};contactSamples={}
    if not sim.getBoolParam(sim.boolparam_headless) then
        demoUI=require 'simUI'
        ui=demoUI.create([[<ui title="双臂装配 · 异常恢复演示" closeable="false" resizable="false" placement="relative" position="-20,40" layout="vbox">
            <label id="1" text="正常装配" style="font-size: 15px; font-weight: bold; min-width: 325px;"/>
            <label id="2" text=""/><label id="3" text=""/>
            <button text="故障模式：持续失败 / 暂时失败" on-click="toggleDemoFaultMode"/>
            <button text="切换：自动恢复 / 保持等待" on-click="toggleDemoRecovery"/>
        </ui>]])
    end
    display()
end

local function skillObservation()
    local skill=demo.retry_skill
    local error
    if skill.kind=='pick' then
        local relative=sim.getObjectPose(blocks[skill.part],arms[skill.arm].tip)
        local binding=skill.metadata.payload_after or skill.payload
        error=dist(relative,binding.offset)
    else error=dist(sim.getObjectPosition(blocks[skill.part],sim.handle_world),cfg.goal[skill.part]) end
    local tracking=0;local actual=qactual(skill.arm)
    for i,v in ipairs(skill.q_end) do tracking=math.max(tracking,math.abs(delta(actual[i],v))) end
    return {position_error_m=error,tracking_error_rad=tracking,
        aligned=error<=demo.retry_config.position_tolerance_m and tracking<.008,
        held_part=held[skill.arm],attempt=retry.attempt}
end

beginAttempt=function(t)
    retry.attempt=retry.attempt+1;retry.start=t;retry.within_since=nil
    retry.blocked=demo.retry_config.failure_mode=='persistent' or retry.attempt<demo.retry_config.max_attempts
    phase='retry'
    baselineBegin(demo.retry_skill,t)
    log('skill_attempt_started',demo.retry_skill,{attempt=retry.attempt,max_attempts=demo.retry_config.max_attempts,
        kind=demo.retry_skill.kind,stroke_limit_injected=retry.blocked})
    display()
end

local function beginRetreat(t,observation)
    local measured=qactual(demo.retry_skill.arm)
    for i,v in ipairs(demo.retry_retreat.q_start) do
        if math.abs(delta(measured[i],v))>.035 then fail('Failed skill is outside the checked withdrawal envelope');return end
    end
    retry.failed_attempts=retry.failed_attempts+1
    if not firstFaultAt then firstFaultAt=t end
    log('skill_attempt_failed',demo.retry_skill,observation)
    states[demo.retry_skill.id]='retrying'
    active[demo.retry_skill.arm]={node=demo.retry_retreat,start=t,within_since=nil}
    retry.retreat_start=t;retry.within_since=nil;phase='retreat'
    log('skill_retreat_started',demo.retry_skill,{attempt=retry.attempt});display()
end

local function retryActuation(t)
    local skill=demo.retry_skill;local side=skill.arm
    pressLoad=nil
    for other,a in pairs(arms) do if other~=side then target(other,a.hold) end end
    if failure then target(side,arms[side].hold);return end
    if phase=='retry' then
        local elapsed=t-retry.start
        local q=plannedQ(skill,elapsed)
        local actual=qactual(side);local tracking=0
        for i,v in ipairs(q) do tracking=math.max(tracking,math.abs(delta(actual[i],v))) end
        peakTracking=math.max(peakTracking,tracking)
        if tracking>.15 and elapsed>.5 then fail('Retry exceeds the original servo tracking limit');return end
        -- Fault emulator limits the final stroke. The retry decision below
        -- uses measured part/joint error, independently of this switch.
        if retry.blocked and elapsed>=demo.retry_block_after_s then q=demo.retry_blocked_q end
        target(side,q)
        local observation=skillObservation()
        if not retry.blocked and skill.kind=='place' and observation.position_error_m<.012 then
            local helper=side=='A' and 'B' or 'A'
            if not support[helper] or not support[helper].ready then fail('Retry installation lacks confirmed support');return end
            pressLoad={part=skill.metadata.support_part or 'beam',force=skill.metadata.force_N or 0,
                       position=skill.metadata.goal or cfg.goal[skill.part]}
        end
        if elapsed>=skill.duration+demo.retry_config.failure_observation_s then
            if observation.aligned then
                -- Physical grasp/snap ownership transfer happens only here.
                baselineFinish(active[side],t)
                if failure then return end
                retry.succeeded=true;phase='normal'
                log('skill_retry_succeeded',skill,observation);display()
            else beginRetreat(t,observation) end
        end
    else
        local node=demo.retry_retreat;local elapsed=t-retry.retreat_start
        target(side,plannedQ(node,elapsed))
        local err=0;local measured=qactual(side)
        for i,v in ipairs(node.q_end) do err=math.max(err,math.abs(delta(measured[i],v))) end
        if elapsed>=node.duration and err<.008 then
            retry.within_since=retry.within_since or t
            if t-retry.within_since>=demo.retry_config.retreat_settle_s then
                active[side]=nil;arms[side].hold={table.unpack(node.q_end)}
                log('skill_retreat_completed',skill,{attempt=retry.attempt,tracking_error_rad=err})
                if retry.attempt<demo.retry_config.max_attempts then beginAttempt(t)
                else
                    states[skill.id]='failed';phase='fault';faultAt=t
                    log('skill_retry_exhausted',skill,{attempts=retry.attempt,snapshot=snapshot()});display()
                end
            end
        else retry.within_since=nil end
        if elapsed>node.duration+8 then fail('Retry withdrawal did not settle') end
    end
end

snapshot=function()
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

beginNode = function(node,t,profileOrigin)
    if phase=='normal' and node.task_id==demo.fault_task then
        interruptedTask=node.task_id
        for _,s in pairs(states) do if s=='done' then prefixCompleted=prefixCompleted+1 end end
        beginAttempt(t)
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
    baselineBegin(node,t,profileOrigin)
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
    phase='recovery';recoveryUsed=true;log('recovery_plan_installed',nil,{from_live_python=hostAccepted,snapshot=snapshot()});display()
end

function sysCall_actuation()
    if phase=='retry' or phase=='retreat' then retryActuation(sim.getSimulationTime());return end
    if phase=='fault' and not failure then
        pressLoad=nil
        for side,a in pairs(arms) do target(side,a.hold) end
        if sim.getSimulationTime()-faultAt>=demo.fault_pause_s and recoveryEnabled and (not liveGate or hostAccepted) then
            activateRecovery()
        end
        return
    end
    baselineActuation()
    if done then phase='done';display() end
    if failure then display() end
end

function sysCall_sensing()
    baselineSensing()
    if pressLoad and stepCount%5==0 then
        contactSamples[#contactSamples+1]={time=sim.getSimulationTime(),load_N=pressLoad.force,
            support_B_N=supportForce('B'),support_B_ready=support.B and support.B.ready or false,
            support_A_N=supportForce('A'),support_A_ready=support.A and support.A.ready or false,phase=phase}
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
    state.phase=phase;state.fault_time=firstFaultAt;state.fault_injected=firstFaultAt~=nil
    state.retry_attempt=retry.attempt;state.failed_attempts=retry.failed_attempts
    state.retry_succeeded=retry.succeeded;state.retry_config=demo.retry_config;state.recovery_used=recoveryUsed
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
