-- Central, predecessor-completion-driven TPG execution on dynamic joint servos.
sim=require 'sim'
local simIK=require 'simIK'
local cfg,program,arms,blocks,nodes,pred,states,active,held,graspSensors,snapSensors,support
local taskTimes,events,trace,done,failure,startTime,finishTime,settled,pressLoad
local peakTracking,peakVelocity,peakSupport,enabled,stepCount,bondViolations
local collisionCollections
local ikEnvironment

local function norm(v) local s=0 for _,x in ipairs(v) do s=s+x*x end return math.sqrt(s) end
local function dist(a,b) return norm({a[1]-b[1],a[2]-b[2],a[3]-b[3]}) end
local function delta(a,b) return math.atan2(math.sin(a-b),math.cos(a-b)) end
local function fail(text)
 if not failure then
  failure=text;sim.addLog(sim.verbosity_errors,'[APEX MR Plus] '..text)
  for _,a in pairs(arms or {}) do
   a.hold={} for i,j in ipairs(a.joints) do a.hold[i]=sim.getJointPosition(j) end
  end
 end
end
local function log(kind,node,details)
 events[#events+1]={time=sim.getSimulationTime(),kind=kind,node=node and node.id or nil,
  task_id=node and node.task_id or nil,arm=node and node.arm or nil,details=details or {}}
end
local function qactual(side)
 local q={} for i,j in ipairs(arms[side].joints) do q[i]=sim.getJointPosition(j) end return q
end
local function target(side,q)
 for i,j in ipairs(arms[side].joints) do sim.setJointTargetPosition(j,q[i]) end
end
local function sensorWorldZ(h)
 local status,f,t=sim.readForceSensor(h)
 if status&1==0 then return nil end
 local m=sim.getObjectMatrix(h,sim.handle_world)
 return m[9]*f[1]+m[10]*f[2]+m[11]*f[3],f,t
end
local function supportForce(side)
 local s=support[side]
 if not s then return 0 end
 local z=sensorWorldZ(arms[side].pad_sensor)
 if not z then return 0 end
 local value=math.abs(z-(s.tare or z))
 s.peak=math.max(s.peak or 0,value);peakSupport=math.max(peakSupport,value)
 return value
end
local function newSensor(name,parent,object,torqueLimit,anchor)
 local h=sim.createForceSensor(0,{0,1,5,0,0},{.006,500,torqueLimit or 100,0,0})
 sim.setObjectAlias(h,name);sim.setObjectParent(h,parent,true)
 local frame=sim.getObjectPose(object,sim.handle_world)
 if anchor then for i=1,3 do frame[i]=anchor[i] end end
 sim.setObjectPose(h,frame,sim.handle_world)
 sim.setObjectInt32Param(h,sim.objintparam_visibility_layer,0)
 sim.setObjectParent(object,h,true)
 sim.resetDynamicObject(object)
 return h
end
local function grasp(node)
 local part=node.part;local side=node.arm;local h=blocks[part]
 if held[side] then fail('Arm already holds '..held[side]);return end
 local relative=sim.getObjectPose(h,arms[side].tip)
 local binding=node.metadata.payload_after or node.payload
 if binding and binding.offset then
  local e=dist(relative,binding.offset)
  if e>.009 then fail(string.format('Grasp alignment %s %.5fm',part,e));return end
 end
 graspSensors[part]=newSensor('GraspFT_'..part,arms[side].palm,h,100)
 held[side]=part
 log('grasp',node,{part=part,dynamic=sim.isDynamicallyEnabled(h),relative_pose=relative})
end
local function releaseAndSnap(node)
 local part=node.part;local side=node.arm;local h=blocks[part]
 if held[side]~=part then fail('Release ownership mismatch '..part);return end
 local definition
 for _,p in ipairs(cfg.parts) do if p.id==part then definition=p end end
 local xyz=sim.getObjectPosition(h,sim.handle_world)
 local error=dist(xyz,definition.goal)
 if error>.009 then fail(string.format('Placement %s error %.5fm',part,error));return end
 local parent=cfg.baseplate
 if part=='beam' then parent=blocks.L3
 elseif part=='cap' then parent=blocks.beam
 elseif #definition.dependencies>0 then parent=blocks[definition.dependencies[1]] end
 -- Keep the measured world pose throughout constraint ownership transfer.
 sim.setObjectParent(h,cfg.root,true)
 if graspSensors[part] then sim.removeObjects({graspSensors[part]});graspSensors[part]=nil end
 local anchor={xyz[1],xyz[2],xyz[3]-definition.size[3]/2}
 if part=='beam' then
  local p=sim.getObjectPosition(blocks.L3,sim.handle_world)
  anchor={p[1],p[2],p[3]+.04}
 end
 snapSensors[part]=newSensor('SnapFT_'..part,parent,h,definition.connection_moment_limit,anchor)
 held[side]=nil
 log('snap_connected',node,{part=part,position_error_m=error,measured_position=xyz,
                          anchor=anchor,model='ideal rigid snap at measured pose'})
end
local function beginNode(node,t)
 local info={node=node,start=t,within_since=nil}
 states[node.id]='running';active[node.arm]=info
 local task=taskTimes[node.task_id]
 if not task then task={id=node.task_id,task_id=node.task_id,arm=node.arm,kind=node.kind,part=node.part,start=t};taskTimes[node.task_id]=task end
 if node.kind=='support_start' then
  local z=sensorWorldZ(arms[node.arm].pad_sensor)
  support[node.arm]={part=node.metadata.support_part or node.part,tare=z,peak=0,ready=false}
  local a=arms[node.arm];local s=support[node.arm]
  -- Compensate the measured, millimetre-scale placement/settling error inside
  -- the contact skill. All IK writes below affect only a virtual IK world.
  simIK.syncFromSim(ikEnvironment,{a.ik_group})
  for i,j in ipairs(a.joints) do simIK.setJointPosition(ikEnvironment,a.ik_map[j],node.q_end[i]) end
  local pose=simIK.getObjectPose(ikEnvironment,a.ik_map[a.tip],simIK.handle_world)
  local beam=blocks[s.part];local m=sim.getObjectMatrix(beam,sim.handle_world)
  local half=cfg.sizes[s.part][3]/2
  local bx,by,bz=m[4]-m[3]*half,m[8]-m[7]*half,m[12]-m[11]*half
  local actualBottom=bz-(m[3]*(pose[1]-bx)+m[7]*(pose[2]-by))/m[11]
  local nominalBottom=cfg.goal[s.part][3]-half
  local dz=actualBottom-nominalBottom
  if math.abs(dz)>.002 then fail(string.format('Support surface outside 2mm skill envelope: %.6fm',dz));return end
  pose[3]=pose[3]+dz
  simIK.setObjectPose(ikEnvironment,a.ik_map[a.ik_target],pose,simIK.handle_world)
  local result=simIK.handleGroup(ikEnvironment,a.ik_group,{syncWorlds=false,allowError=false})
  if result~=simIK.result_success then fail('Virtual support-height IK failed');return end
  s.dq={};local maxdq=0
  for i,j in ipairs(a.joints) do
   s.dq[i]=delta(simIK.getJointPosition(ikEnvironment,a.ik_map[j]),node.q_end[i]);maxdq=math.max(maxdq,math.abs(s.dq[i]))
  end
  if maxdq>.015 then fail('Support correction changed IK branch');return end
  s.height_correction_m=dz
  log('support_height_compensation',node,{height_m=dz,max_joint_correction_rad=maxdq,virtual_ik_only=true})
  sim.setObjectInt32Param(arms[node.arm].pad,sim.shapeintparam_respondable,1)
  -- The physics engine caches respondability when a body is created.
  sim.resetDynamicObject(arms[node.arm].pad)
  sim.setObjectInt32Param(arms[node.arm].pad,sim.objintparam_visibility_layer,1)
 end
 log('node_started',node)
end
local function plannedQ(node,elapsed)
 local segments=node.motion_segments or {}
 local t=elapsed
 for _,s in ipairs(segments) do
  if t<=s.duration then
   local u=math.max(0,math.min(1,t/s.duration));local v=u*u*u*(10+u*(-15+6*u))
   local q={} for i=1,6 do q[i]=s.q0[i]+(s.q1[i]-s.q0[i])*v end return q
  end
  t=t-s.duration
 end
 -- The contact controller adds a small correction to its local target.
 -- Never return a graph-owned array that could accumulate that correction.
 return {table.unpack(node.q_end or node.q_samples[#node.q_samples])}
end
local function finishNode(info,t)
 local node=info.node;local kind=node.kind
 if kind=='pick' or node.metadata.action=='grasp' then grasp(node) end
 if kind=='place' or node.metadata.action=='release' then
  if (node.metadata.force_N or 0)>0 then
   local helper=node.arm=='A' and 'B' or 'A'
   if not support[helper] or not support[helper].ready or (support[helper].peak or 0)<.4 then
    fail('No measured support reaction during insertion '..node.part);return
   end
  end
  releaseAndSnap(node)
 end
 if kind=='support_start' then
  support[node.arm].ready=true
  log('support_ready',node,{force_N=supportForce(node.arm),tare_world_z_N=support[node.arm].tare})
 elseif kind=='support_end' then
  log('support_released',node,{peak_normal_N=support[node.arm] and support[node.arm].peak or 0})
  sim.setObjectInt32Param(arms[node.arm].pad,sim.shapeintparam_respondable,0)
  sim.resetDynamicObject(arms[node.arm].pad)
  sim.setObjectInt32Param(arms[node.arm].pad,sim.objintparam_visibility_layer,0)
  support[node.arm]=nil
 end
 if failure then return end
 states[node.id]='done';active[node.arm]=nil
 arms[node.arm].hold={table.unpack(node.q_end)}
 if kind=='support_start' and support[node.arm] and support[node.arm].dq then
  for i=1,6 do arms[node.arm].hold[i]=arms[node.arm].hold[i]+support[node.arm].dq[i] end
 end
 local tm=taskTimes[node.task_id];tm['end']=t;tm.duration=t-tm.start
 log('node_completed',node)
end

function sysCall_init()
 cfg=sim.readCustomTableData(sim.getObject('/APEX_MR_Plus'),'NormalAssembly')
 cfg.root=sim.getObject('/APEX_MR_Plus');cfg.baseplate=sim.getObject('/APEX_MR_Plus/CastleBaseplate')
 program=cfg.program;assert(program and program.nodes and #program.nodes>0,'No TPG program embedded')
 arms={};blocks={};nodes={};pred={};states={};active={};held={};graspSensors={};snapSensors={};support={}
 taskTimes={};events={};trace={};done=false;failure=nil;settled=false;pressLoad=nil;stepCount=0;bondViolations={}
 peakTracking=0;peakVelocity=0;peakSupport=0;enabled={};startTime=sim.getSimulationTime()
 ikEnvironment=simIK.createEnvironment()
 for _,side in ipairs({'A','B'}) do
  local model=sim.getObject('/APEX_MR_Plus/'..side..'_UR5')
  local a={model=model,joints={}}
  for i=1,6 do a.joints[i]=sim.getObject(':/'..side..'_joint_'..i,{proxy=model}) end
  a.tip=sim.getObject(':/'..side..'_TCP',{proxy=model})
  a.palm=sim.getObject(':/'..side..'_GripperPalm',{proxy=model})
  a.pad=sim.getObject(':/'..side..'_SupportShoe',{proxy=model})
  a.pad_sensor=sim.getObject(':/'..side..'_PadForceSensor',{proxy=model})
  a.tool_sensor=sim.getObject(':/'..side..'_ToolForceSensor',{proxy=model})
  a.ik_target=sim.getObject('/APEX_MR_Plus/'..side..'_IK_Target')
  a.ik_group=simIK.createGroup(ikEnvironment)
  simIK.setGroupCalculation(ikEnvironment,a.ik_group,simIK.method_damped_least_squares,.02,60)
  local element,mapping=simIK.addElementFromScene(ikEnvironment,a.ik_group,model,a.tip,a.ik_target,simIK.constraint_pose)
  a.ik_map=mapping
  a.collision=sim.createCollection(1)
  for _,h in ipairs(sim.getObjectsInTree(model,sim.object_shape_type,0)) do
   local alias=sim.getObjectAlias(h)
   local physical=h==model or sim.getObjectInt32Param(h,sim.shapeintparam_static)==0 or
    sim.getObjectInt32Param(h,sim.shapeintparam_respondable)~=0 or
    string.match(alias,'_GripperPalm') or string.match(alias,'_Finger') or string.match(alias,'_SupportShoe')
   if physical then sim.addItemToCollection(a.collision,sim.handle_single,h,0) end
  end
  arms[side]=a;a.hold=qactual(side)
  sim.setObjectInt32Param(a.pad,sim.shapeintparam_respondable,0)
 end
 for _,p in ipairs(cfg.parts) do blocks[p.id]=sim.getObject('/APEX_MR_Plus/'..p.id) end
 for _,node in ipairs(program.nodes) do nodes[node.id]=node;pred[node.id]={};states[node.id]='pending' end
 for _,edge in ipairs(program.edges) do pred[edge.target][#pred[edge.target]+1]=edge.source end
end

function sysCall_actuation()
 local t=sim.getSimulationTime()
 pressLoad=nil
 if failure or done then for side,a in pairs(arms) do target(side,a.hold) end return end
 if t-startTime<.6 then for side,a in pairs(arms) do target(side,a.hold) end return end
 if not settled then
  settled=true
  for side,a in pairs(arms) do
   enabled[side]={joints={},tool_sensor=sim.isDynamicallyEnabled(a.tool_sensor),pad_sensor=sim.isDynamicallyEnabled(a.pad_sensor),pad=sim.isDynamicallyEnabled(a.pad)}
   for i,j in ipairs(a.joints) do enabled[side].joints[i]=sim.isDynamicallyEnabled(j);if not enabled[side].joints[i] then fail('Joint dynamics disabled '..side..i) end end
  end
  log('execution_ready',nil,{dynamic_enablement=enabled})
 end
 -- Nodes are admitted only when every graph predecessor has completed.
 for _,node in ipairs(program.nodes) do
  if states[node.id]=='pending' and not active[node.arm] then
   local ready=true for _,p in ipairs(pred[node.id]) do if states[p]~='done' then ready=false;break end end
   if ready then beginNode(node,t) end
  end
 end
 for _,side in ipairs({'A','B'}) do
  local info=active[side]
  if info then
   local node=info.node;local elapsed=t-info.start
   local q=plannedQ(node,elapsed)
   if support[side] and support[side].dq and (node.kind=='support_start' or node.kind=='support_end') then
    local motion=0 for _,seg in ipairs(node.motion_segments or {}) do motion=motion+seg.duration end
    local u=math.max(0,math.min(1,elapsed/math.max(.08,motion)))
    local blend=u*u*u*(10+u*(-15+6*u))
    if node.kind=='support_end' then blend=1-blend end
    for i=1,6 do q[i]=q[i]+blend*support[side].dq[i] end
   end
   target(side,q)
   local actual=qactual(side);local err=0
   for i=1,6 do err=math.max(err,math.abs(delta(actual[i],q[i]))) end
   peakTracking=math.max(peakTracking,err)
   if err>.15 and elapsed>.5 then fail(string.format('Servo tracking %.4frad on %s',err,side)) end
   if node.kind=='place' and (node.metadata.force_N or 0)>0 then
    local helper=side=='A' and 'B' or 'A'
    if not support[helper] or not support[helper].ready then fail('Insertion lacks support-ready barrier')
    else pressLoad={part=node.metadata.support_part or 'beam',force=node.metadata.force_N,position=node.metadata.goal or cfg.goal[node.part]} end
   end
   if elapsed>=node.duration then
    local qualified=err<.008
    if node.kind=='support_start' then qualified=qualified and supportForce(side)>.15 end
    if qualified then
     info.within_since=info.within_since or t
     local extra=0.04
     if node.node_type=='skill' then extra=node.metadata.dwell_s or .20 end
     if t-info.within_since>=extra then finishNode(info,t) end
    else info.within_since=nil end
   end
   if elapsed>node.duration+8 then fail('Endpoint did not settle: '..node.id) end
  else target(side,arms[side].hold) end
 end
 local all=true for _,state in pairs(states) do if state~='done' then all=false;break end end
 if all then done=true;finishTime=t;log('assembly_completed',nil) end
end

function sysCall_dyn(inData)
 if not inData.afterStep and pressLoad and not failure then
  local h=blocks[pressLoad.part];local p=sim.getObjectPosition(h,sim.handle_world)
  local rx=pressLoad.position[1]-p[1];local ry=pressLoad.position[2]-p[2];local f=pressLoad.force
  -- Bullet substeps share the external-force accumulator. Reset it before
  -- each 8 N command so ten substeps do not accidentally produce 8..80 N.
  sim.addForceAndTorque(h|sim.handleflag_resetforcetorque,{0,0,-f},{-ry*f,rx*f,0})
 end
end

function sysCall_sensing()
 stepCount=stepCount+1
 for side,a in pairs(arms) do
  for _,j in ipairs(a.joints) do peakVelocity=math.max(peakVelocity,math.abs(sim.getJointVelocity(j))) end
  if support[side] then supportForce(side) end
 end
 if stepCount%5==0 then
  local hit,pair=sim.checkCollision(arms.A.collision,arms.B.collision)
  if hit>0 then fail('Actual inter-arm collision: '..sim.getObjectAlias(pair[1],1)..' / '..sim.getObjectAlias(pair[2],1)) end
 end
 -- Explicit, measured snap overload monitoring (no undocumented broken bit).
 for part,h in pairs(snapSensors) do
  local status,f,t=sim.readForceSensor(h)
  if status&1~=0 then
   local limit=1.2
   bondViolations[part]=norm(t)>limit and (bondViolations[part] or 0)+1 or 0
   if bondViolations[part]>=10 then fail('Snap torque limit exceeded at '..part) end
  end
 end
 if stepCount%5==0 then
  local row={time=sim.getSimulationTime(),qA=qactual('A'),qB=qactual('B'),parts={},support={},snap={},
             applied_insertion_load_N=pressLoad and pressLoad.force or 0}
  for name,h in pairs(blocks) do row.parts[name]=sim.getObjectPose(h,sim.handle_world) end
  for name,h in pairs(snapSensors) do
   local status,force,torque=sim.readForceSensor(h)
   row.snap[name]={valid=status&1~=0}
   if status&1~=0 then
    row.snap[name].force_N=force;row.snap[name].torque_Nm=torque;row.snap[name].torque_norm_Nm=norm(torque)
   end
  end
  for side,a in pairs(arms) do
   local z,f,torque=sensorWorldZ(a.tool_sensor)
   row[side..'_tool_force_N']=f;row[side..'_tool_torque_Nm']=torque
   row[side..'_joint_torque_Nm']={};row[side..'_joint_velocity_rad_s']={}
   for i,j in ipairs(a.joints) do
    row[side..'_joint_torque_Nm'][i]=sim.getJointForce(j)
    row[side..'_joint_velocity_rad_s'][i]=sim.getJointVelocity(j)
   end
   row.support[side]=supportForce(side)
  end
  trace[#trace+1]=row
 end
end

function getState()
 local count=0 for _,s in pairs(states or {}) do if s=='done' then count=count+1 end end
 return {done=done,failure=failure,time=sim.getSimulationTime(),completed=count,total=program and #program.nodes or 0,
         peak_tracking_rad=peakTracking,peak_velocity_rad_s=peakVelocity,peak_support_normal_N=peakSupport,
         active_A=active.A and active.A.node.id,active_B=active.B and active.B.node.id}
end
function getResult()
 local result=getState();result.events=events;result.trace=trace;result.dynamic_enablement=enabled
 result.node_states=states;result.task_schedule={};result.part_poses={};result.makespan=(finishTime or sim.getSimulationTime())-startTime
 for _,tm in pairs(taskTimes) do result.task_schedule[#result.task_schedule+1]=tm end
 for name,h in pairs(blocks) do result.part_poses[name]=sim.getObjectPose(h,sim.handle_world) end
 result.snap_dynamic={};for part,h in pairs(snapSensors) do result.snap_dynamic[part]=sim.isDynamicallyEnabled(h) end
 return result
end
function sysCall_cleanup()
 for _,a in pairs(arms or {}) do if a.collision then sim.destroyCollection(a.collision) end end
 if ikEnvironment then simIK.eraseEnvironment(ikEnvironment) end
end
