-- Task-driven profile. This file is embedded after the copied controller.lua.
-- Bodies move only through joint servo targets, gravity and measured-pose
-- force-sensor constraints. It never sets a body pose or a joint position.
SkillnetV13={}
local K,H
local function clone(value)
 if type(value)~='table' then return value end
 local result={} for key,item in pairs(value) do result[key]=clone(item) end return result
end
local function emit(kind,node,details)
 K.events[#K.events+1]={kind=kind,time=sim.getSimulationTime(),node=node and node.id,
  task_id=node and node.task_id,arm=node and node.arm,details=details or {}}
end
local function joints(side)
 local q={} for i,j in ipairs(K.arms[side].joints) do q[i]=sim.getJointPosition(j) end return q
end
local function command(side,q)
 for i,j in ipairs(K.arms[side].joints) do sim.setJointTargetPosition(j,q[i]) end
end
local function abort(reason,node,kind,details)
 if K.failure then return end
 K.failure=reason;emit(kind or 'execution_failure',node,details or {reason=reason})
 if kind=='collision_failure' then K.collision_failures[#K.collision_failures+1]=details or {reason=reason} end
 if kind=='tracking_failure' then K.tracking_failures[#K.tracking_failures+1]=details or {reason=reason} end
 for side,arm in pairs(K.arms) do arm.hold=joints(side) end
 sim.addLog(sim.verbosity_errors,'[Skillnet v1.3] '..reason)
end
local function resolve(value)
 if type(value)=='number' then return value end
 local handle=K.aliases[value]
 assert(handle,'Missing task object: '..tostring(value))
 return handle
end
local function collection(shapes)
 local handle=sim.createCollection(1)
 for _,shape in ipairs(shapes) do sim.addItemToCollection(handle,sim.handle_single,shape,0) end
 K.collections[#K.collections+1]=handle
 return handle
end
local function actionOf(node)
 if node.node_type=='pose' then return nil end
 return (node.metadata or {}).action
end
local function bindingCheck(node,binding)
 if not binding then abort('Missing grasp binding',node);return false end
 local body=K.blocks[node.part]
 local measured=sim.getObjectPose(body,K.arms[node.arm].tip)
 local wanted={binding.offset[1],binding.offset[2],binding.offset[3],table.unpack(binding.quaternion)}
 local ep,er=H.dist(measured,wanted),H.orientationError(measured,wanted)
 if ep>(K.runtime.grasp_tolerance_m or .009) or er>(K.runtime.grasp_orientation_tolerance_rad or .06) then
  abort('Measured grasp alignment outside tolerance: '..node.part,node,nil,
   {position_error_m=ep,orientation_error_rad=er});return false
 end
 return true
end
local function placementCheck(node)
 local wanted=(node.metadata or {}).target_pose or (K.cfg.goal_poses or {})[node.part]
 if not wanted then abort('Missing placement pose: '..tostring(node.part),node);return false end
 local measured=sim.getObjectPose(K.blocks[node.part],sim.handle_world)
 local ep,er=H.dist(measured,wanted),H.orientationError(measured,wanted)
 if ep>(K.runtime.pose_tolerance_m or .005) or er>(K.runtime.orientation_tolerance_rad or .02) then
  abort('Measured placement outside tolerance: '..node.part,node,nil,
   {position_error_m=ep,orientation_error_rad=er,measured=measured,wanted=wanted});return false
 end
 return true,ep,er,measured
end
local function releaseGrip(node)
 local part,side=node.part,node.arm
 if K.held[side]~=part or not K.grips[part] then abort('Release ownership mismatch',node);return false end
 sim.setObjectParent(K.blocks[part],K.cfg.root,true)
 sim.removeObjects({K.grips[part]});K.grips[part]=nil
 sim.resetDynamicObject(K.blocks[part]);K.held[side]=nil
 return true
end
local function supportForce(side)
 local arm=K.arms[side]
 local z=H.sensorWorldZ(arm.pad_sensor)
 if z==nil or arm.pad_tare==nil then return {valid=false,force_N=0,ready=false} end
 local force=math.abs(z-arm.pad_tare)
 local required=(K.cfg.validation_requirements or {}).support_force_min_N or 1.
 local minimum=math.max(1.,required,K.runtime.support_force_min_N or 1.)
 return {valid=true,force_N=force,ready=force>=minimum,threshold_N=minimum}
end
local function expireEndpointContacts(part)
 if not part then return end
 for side,receipt in pairs(K.endpoint_contacts) do
  local expire=receipt.part==part
  for _,contact in ipairs(receipt.contacts) do if contact.lower==part then expire=true end end
  if expire then K.endpoint_contacts[side]=nil end
 end
end
local function retainEndpointContacts(node)
 local side,part=node.arm,node.part
 local meta=node.metadata or {}
 if node.node_type~='pose' or meta.request_kind~='linear' or
    not part or K.held[side]~=part or not K.grips[part] then return end
 local contacts={}
 local function partName(value)
  for name,handle in pairs(K.blocks) do if name==value or handle==value then return name end end
 end
 for _,pair in ipairs(meta.contact_exclusions or {}) do
  local first,second=partName(pair[1]),partName(pair[2])
  local lower=first==part and second or second==part and first
  if lower and K.installed[lower] and not K.grips[lower] then
   contacts[#contacts+1]={pair=clone(pair),lower=lower,
    slot=K.installed[lower],snap=K.snaps[lower]}
  end
 end
 if #contacts>0 then
  K.endpoint_contacts[side]={part=part,grip=K.grips[part],q=clone(node.q_end),contacts=contacts,
   source_task=node.task_id or node.id,source_node=node.id}
 end
end
local function finishAction(node,supportMeasurement)
 local action,meta=actionOf(node),node.metadata or {}
 local side,part=node.arm,node.part
 if action=='grasp' or action=='release' or action=='release_buffer' or
    action=='install' or action=='detach' then expireEndpointContacts(part) end
 if action=='grasp' then
  if K.held[side] then abort('Grasp requires empty receiving arm',node);return end
  if not K.blocks[part] then abort('Unknown grasp part '..tostring(part),node);return end
  local binding=meta.transfer_payload or meta.payload_after
  if not bindingCheck(node,binding) then return end
  if meta.handover then
   local giver=meta.giver
   if K.held[giver]~=part or not K.grips[part] then abort('Handover giver ownership mismatch',node);return end
   if K.transfers[part] then abort('Duplicate handover receive',node);return end
   local old=K.grips[part]
   K.grips[part]=H.newSensor('TransferFT_'..part,K.arms[side].palm,K.blocks[part],100.)
   K.transfers[part]={giver=giver,taker=side,old_sensor=old}
   K.held[side]=part
   emit('handover_received',node,{part=part,giver=giver,taker=side,
    relative_pose=sim.getObjectPose(K.blocks[part],K.arms[side].tip),dynamic=sim.isDynamicallyEnabled(K.blocks[part])})
  else
   K.grips[part]=H.newSensor('GraspFT_'..part,K.arms[side].palm,K.blocks[part],100.)
   K.held[side]=part
   emit('grasp',node,{part=part,dynamic=sim.isDynamicallyEnabled(K.blocks[part]),
    relative_pose=sim.getObjectPose(K.blocks[part],K.arms[side].tip)})
  end
 elseif action=='release' or action=='release_buffer' then
  if meta.handover then
   local transfer=K.transfers[part]
   if not transfer or transfer.giver~=side or K.held[transfer.taker]~=part then
    abort('Handover release before receiver grasp',node);return
   end
   sim.removeObjects({transfer.old_sensor});K.held[side]=nil;K.transfers[part]=nil
   emit('handover_released',node,{part=part,giver=side,taker=transfer.taker,rigid_snap=false})
  else
   local ok,ep,er=placementCheck(node);if not ok or not releaseGrip(node) then return end
   K.placed[part]=meta.location or false
   emit('release',node,{part=part,location=meta.location,rigid_snap=false,
    position_error_m=ep,orientation_error_rad=er})
  end
 elseif action=='install' then
  local ok,ep,er,measured=placementCheck(node);if not ok then return end
  for supportArm,slot in pairs(K.supporting) do
   if slot==meta.slot then
    local measurement=supportForce(supportArm)
    if not measurement.ready then abort('Install without measured support load',node);return end
    emit('support_load_verified',node,{part=part,arm=supportArm,slot=slot,
     valid=measurement.valid,force_N=measurement.force_N})
   end
  end
  if not releaseGrip(node) then return end
  local parent=K.cfg.baseplate
  local lower=(meta.below or {})[1]
  if lower then
   parent=nil
   for installed,slot in pairs(K.installed) do if slot==lower then parent=K.blocks[installed];break end end
   if not parent then abort('Installed lower slot missing: '..lower,node);return end
  end
  local anchor=clone(measured);anchor[3]=anchor[3]-(K.definitions[part].size[3]/2)
  K.snaps[part]=H.newSensor('SnapFT_'..part,parent,K.blocks[part],10.,anchor)
  K.installed[part]=meta.slot;K.placed[part]=meta.slot
  emit('part_installed',node,{part=part,slot=meta.slot,parent=sim.getObjectAlias(parent),
   position_error_m=ep,orientation_error_rad=er,dynamic=sim.isDynamicallyEnabled(K.blocks[part]),
   constraint_dynamic=sim.isDynamicallyEnabled(K.snaps[part]),rigid_snap=true})
 elseif action=='detach' then
  if K.held[side]~=part or not K.grips[part] or not K.installed[part] or not K.snaps[part] then
   abort('Detach requires installed object held by gripper',node);return
  end
  -- Grasp already moved the body under the measured-pose gripper constraint.
  -- Removing the now empty assembly sensor cannot move the body.
  sim.removeObjects({K.snaps[part]});K.snaps[part]=nil;K.installed[part]=nil;K.placed[part]=nil
  sim.setObjectInt32Param(K.blocks[part],sim.shapeintparam_static,0)
  sim.setObjectInt32Param(K.blocks[part],sim.shapeintparam_respondable,1)
  sim.resetDynamicObject(K.blocks[part])
  emit('detach',node,{part=part,slot=meta.slot,dynamic=sim.isDynamicallyEnabled(K.blocks[part]),gripper_retained=true})
 elseif action=='support_start' then
  local measurement=supportMeasurement or supportForce(side)
  K.supporting[side]=meta.slot
  emit('support_started',node,{slot=meta.slot,sensor_ready=K.arms[side].pad_tare~=nil,
   valid=measurement.valid,force_N=measurement.force_N,measurement=measurement,
   force_qualification='current measured load qualified at readiness'})
 elseif action=='support_end' then
  if K.supporting[side]~=meta.slot then abort('Support release ownership mismatch',node);return end
  emit('support_ended',node,{slot=meta.slot,measurement=supportForce(side)})
  K.supporting[side]=nil
 elseif action~=nil and action~='wait' and action~='hold' then
  abort('Unsupported generic skill action '..tostring(action),node)
 end
end
local function ready(node)
 for _,previous in ipairs(K.pred[node.id]) do if K.states[previous]~='done' then return false end end
 return true
end
local function loadContactComplete(node)
 for _,task in ipairs((node.metadata or {}).support_load_contact_tasks or {}) do
  for _,ident in ipairs(K.task_nodes[task]) do if K.states[ident]~='done' then return false end end
 end
 return true
end
local function continuationPassed(node,actual,planned)
 -- Match the archived waypointPassed geometry: actual along-path progress
 -- acknowledges a milestone; transverse error retains endpoint tolerance.
 local segment=(node.motion_segments or {})[1]
 if not segment or not segment.profile_duration then return false end
 local unwrapped={}
 for i=1,6 do unwrapped[i]=planned[i]+H.delta(actual[i],planned[i]) end
 local dot,length=0,0
 for i=1,6 do
  local d=segment.progress_direction and segment.progress_direction[i] or (segment.q1[i]-segment.q0[i])
  dot=dot+(unwrapped[i]-node.q_end[i])*d;length=length+d*d
 end
 if length<1e-15 or dot/length < -1e-8 then return false end
 local error=0
 for i=1,6 do
  local d=segment.progress_direction and segment.progress_direction[i] or (segment.q1[i]-segment.q0[i])
  local onLine=node.q_end[i]+dot/length*d
  error=math.max(error,math.abs(unwrapped[i]-onLine))
 end
 return error<(K.runtime.endpoint_tolerance_rad or .008)
end
local function begin(node,t,origin)
 K.endpoint_contacts[node.arm]=nil
 local meta=node.metadata or {}
 if actionOf(node)=='support_start' then
  local arm=K.arms[node.arm]
  sim.setObjectInt32Param(arm.pad,sim.shapeintparam_respondable,1)
  sim.setObjectInt32Param(arm.pad,sim.objintparam_visibility_layer,1)
  sim.resetDynamicObject(arm.pad)
  arm.pad_tare=H.sensorWorldZ(arm.pad_sensor)
 end
 K.states[node.id]='running'
 K.active[node.arm]={node=node,start=t,profile_origin=origin or (t-(meta.motion_run_offset or 0))}
 local key=node.task_id or node.id
 if not K.times[key] then K.times[key]={id=key,task_id=key,arm=node.arm,kind=node.kind,part=node.part,start=t} end
 emit('node_started',node)
end
local function dispatch(side,t)
 if K.active[side] then return end
 for _,node in ipairs(K.program.nodes) do
  if node.arm==side and K.states[node.id]=='pending' and ready(node) then begin(node,t);return end
 end
end
local function complete(info,t,supportMeasurement)
 local node=info.node
 finishAction(node,supportMeasurement);if K.failure then return end
 K.states[node.id]='done';K.active[node.arm]=nil;K.arms[node.arm].hold=clone(node.q_end)
 retainEndpointContacts(node)
 local task=K.times[node.task_id or node.id];task['end']=t;task.duration=t-task.start
 emit('node_completed',node)
end
local function qualified(node)
 local action,meta=actionOf(node),node.metadata or {}
 if action=='support_start' then
  -- Contact approach may precede the load. Readiness itself requires a real
  -- current load; the existing dwell resets whenever this condition is lost.
  local measurement=supportForce(node.arm)
  return measurement.valid and measurement.ready,measurement
 end
 if action=='install' then
  for side,slot in pairs(K.supporting) do
   if slot==meta.slot and not supportForce(side).ready then return false end
  end
 end
 return true
end
function SkillnetV13.init(cfg,helpers)
 H=helpers
 K={cfg=cfg,runtime=cfg.runtime or {},program=cfg.program,arms={},blocks={},definitions={},
  aliases={},collections={},nodes={},task_nodes={},pred={},states={},active={},held={},grips={},snaps={},endpoint_contacts={},
  installed={},supporting={},placed={},transfers={},times={},events={},trace={},
  collision_failures={},tracking_failures={},start=sim.getSimulationTime(),step=0,
  collision_checks=0,tracking_samples=0,peak_tracking=0,peak_velocity=0,done=false}
 assert(K.program and K.program.nodes and #K.program.nodes>0,'No skillnet TPG embedded')
 for _,handle in ipairs(sim.getObjectsInTree(cfg.root,sim.handle_all,0)) do
  K.aliases[sim.getObjectAlias(handle)]=handle
 end
 K.fixedInstallation={}
 for _,pair in ipairs(cfg.fixed_installation_contacts or {}) do
  local a,b=resolve(pair[1]),resolve(pair[2])
  for _,h in ipairs({a,b}) do
   assert(sim.getObjectParent(h)==cfg.root and sim.getObjectType(h)==sim.object_shape_type
    and sim.getObjectInt32Param(h,sim.shapeintparam_static)~=0,'Installation contact requires fixed root shapes')
  end
  K.fixedInstallation[a]=K.fixedInstallation[a] or {}
  K.fixedInstallation[b]=K.fixedInstallation[b] or {}
  K.fixedInstallation[a][b]=true;K.fixedInstallation[b][a]=true
 end
 for _,part in ipairs(cfg.parts) do K.blocks[part.id]=resolve(part.alias or part.id);K.definitions[part.id]=part end
 for part,handle in pairs(cfg.initial_attachments or {}) do K.snaps[part]=resolve(handle) end
 for slot,part in pairs(cfg.initial_installed or {}) do K.installed[part]=slot end
 K.enabled={objects={}}
 for side,old in pairs(cfg.arms) do
  local arm={model=resolve(old.model),joints=clone(old.joints),tip=resolve(old.tip),palm=resolve(old.palm),
   pad=resolve(old.support_pad),pad_sensor=resolve(old.support_sensor),tool_sensor=resolve(old.sensor),
   shapes={},links={},home_q=clone(old.home_q)}
  K.arms[side]=arm;arm.hold=joints(side)
  local jointIndex={} for index,joint in ipairs(arm.joints) do jointIndex[joint]=index end
  for _,handle in ipairs(sim.getObjectsInTree(arm.model,sim.object_shape_type,0)) do
   local alias=sim.getObjectAlias(handle)
   local explicit=alias:match('^'..side..'_') and (alias:find('Palm') or alias:find('Finger') or alias:find('SupportShoe'))
   if handle==arm.model or sim.getObjectInt32Param(handle,sim.shapeintparam_static)==0 or
    sim.getObjectInt32Param(handle,sim.shapeintparam_respondable)~=0 or explicit then
    arm.shapes[#arm.shapes+1]=handle
    local parent,index=handle,0
    while parent~=-1 and parent~=cfg.root do
     if jointIndex[parent] then index=jointIndex[parent];break end
     parent=sim.getObjectParent(parent)
    end
    arm.links[index]=arm.links[index] or {};table.insert(arm.links[index],handle)
   end
  end
  arm.collision=collection(arm.shapes);arm.linkCollections={};arm.selfpairs={}
  for index,shapes in pairs(arm.links) do arm.linkCollections[index]=collection(shapes) end
  for i,one in pairs(arm.linkCollections) do for j,two in pairs(arm.linkCollections) do
   if j>i+1 then arm.selfpairs[#arm.selfpairs+1]={one,two} end
  end end
  K.enabled[side]={joints={},tool_sensor=sim.isDynamicallyEnabled(arm.tool_sensor),pad_sensor=sim.isDynamicallyEnabled(arm.pad_sensor)}
  for i,joint in ipairs(arm.joints) do K.enabled[side].joints[i]=sim.isDynamicallyEnabled(joint) end
 end
 for _,node in ipairs(K.program.nodes) do
  K.nodes[node.id]=node;K.pred[node.id]={};K.states[node.id]='pending'
  local task=node.task_id or node.id
  K.task_nodes[task]=K.task_nodes[task] or {};table.insert(K.task_nodes[task],node.id)
 end
 for _,node in ipairs(K.program.nodes) do
  for _,task in ipairs((node.metadata or {}).support_load_contact_tasks or {}) do
   assert(K.task_nodes[task],'Missing load-contact task: '..tostring(task))
  end
 end
 for _,edge in ipairs(K.program.edges) do table.insert(K.pred[edge.target],edge.source) end
 for part,handle in pairs(K.blocks) do K.enabled.objects[part]=sim.isDynamicallyEnabled(handle) end
 emit('controller_initialized',nil,{profile='skillnet_v1_3'})
end
function SkillnetV13.actuation()
 local t=sim.getSimulationTime()
 if K.failure or K.done then for side,arm in pairs(K.arms) do command(side,arm.hold) end return end
 if t-K.start<(K.runtime.initial_settle_s or .6) then
  for side,arm in pairs(K.arms) do command(side,arm.hold) end return
 end
 if not K.enablement_checked then
  for side,arm in pairs(K.arms) do
   K.enabled[side].tool_sensor=sim.isDynamicallyEnabled(arm.tool_sensor)
   K.enabled[side].pad_sensor=sim.isDynamicallyEnabled(arm.pad_sensor)
   for i,joint in ipairs(arm.joints) do
    K.enabled[side].joints[i]=sim.isDynamicallyEnabled(joint)
    if not K.enabled[side].joints[i] then abort('Dynamic joint disabled: '..side..i);return end
   end
   if not K.enabled[side].tool_sensor or not K.enabled[side].pad_sensor then abort('Dynamic tool sensor disabled: '..side);return end
  end
  for part,body in pairs(K.blocks) do
   K.enabled.objects[part]=sim.isDynamicallyEnabled(body)
   if not K.enabled.objects[part] then abort('Dynamic task body disabled: '..part);return end
  end
  K.enablement_checked=true
  emit('execution_ready',nil,{profile='skillnet_v1_3',dynamic_enablement=K.enabled,
   collision_monitor=true,tracking_monitor=true})
 end
 for _,side in ipairs({'A','B'}) do dispatch(side,t) end
 for _,side in ipairs({'A','B'}) do
  local transitions=0
  while K.active[side] do
   local info=K.active[side];local node=info.node;local meta=node.metadata or {}
   local elapsed=meta.motion_run_id and (t-info.profile_origin-(meta.motion_run_offset or 0)) or (t-info.start)
   local planned=H.plannedQ(node,elapsed);command(side,planned)
   local actual=joints(side);local error=0
   for i=1,6 do error=math.max(error,math.abs(H.delta(actual[i],planned[i]))) end
   K.tracking_samples=K.tracking_samples+1;K.peak_tracking=math.max(K.peak_tracking,error)
   if error>(K.runtime.tracking_limit_rad or .15) and elapsed>.5 then
    abort('Servo tracking limit on '..side,node,'tracking_failure',{arm=side,error_rad=error});return
   end
   local finished=false
   if meta.continuation_to then
    if elapsed>=node.duration and continuationPassed(node,actual,planned) then
     local nextNode=K.nodes[meta.continuation_to];local origin=info.profile_origin
     complete(info,t);if K.failure then return end
     if not nextNode or not ready(nextNode) then abort('Continuous motion dependency not acquired',node);return end
     begin(nextNode,t,origin);finished=true
    end
   else
    local physicalReady,supportMeasurement=qualified(node)
    local settled=error<(K.runtime.endpoint_tolerance_rad or .008) and physicalReady
    if node.node_type~='skill' and elapsed<node.duration then settled=false end
    if settled then info.within_since=info.within_since or t else info.within_since=nil end
    local dwell=node.node_type=='skill' and (meta.qualified_dwell_s or meta.dwell_s or .20) or .04
    if elapsed>=node.duration and info.within_since and t-info.within_since>=dwell then
     complete(info,t,supportMeasurement);finished=true;if K.failure then return end;dispatch(side,t)
    end
   end
   local guardElapsed=elapsed-node.duration
   if actionOf(node)=='support_start' and #(meta.support_load_contact_tasks or {})>0 and
      error<(K.runtime.endpoint_tolerance_rad or .008) then
    -- Reaching the shoe pose and waiting for another arm's contact are distinct
    -- from a failed endpoint. The original force-guard allowance starts only
    -- once that compiled contact path has actually completed. Joint tracking,
    -- endpoint acquisition and the executor's shared deadline remain enforced.
    if loadContactComplete(node) then info.load_contact_at=info.load_contact_at or t end
    guardElapsed=info.load_contact_at and (t-info.load_contact_at) or 0.
   end
   if guardElapsed>(K.runtime.endpoint_timeout_s or 10.) then abort('Endpoint or physical guard timeout: '..node.id,node);return end
   transitions=transitions+1;if not finished or transitions>#K.program.nodes then break end
  end
  if not K.active[side] then command(side,K.arms[side].hold) end
 end
 local all=true for _,state in pairs(K.states) do if state~='done' then all=false;break end end
 if all then
  K.complete_since=K.complete_since or t
  if t-K.complete_since>=(K.runtime.final_settle_s or .5) then
   if K.held.A or K.held.B or next(K.supporting) or next(K.transfers) then abort('TPG finished with occupied arms');return end
   K.done=true;K.finish=t;emit('assembly_completed',nil)
  end
 end
end
local function isContact(first,second)
 local alias1,alias2=sim.getObjectAlias(first),sim.getObjectAlias(second)
 local function matches(pair)
  local a,b=pair[1],pair[2]
  return (a==first or a==alias1) and (b==second or b==alias2) or
   (a==second or a==alias2) and (b==first or b==alias1)
 end
 local allowed=false
 local endpointReceipts={}
 for _,info in pairs(K.active) do for _,pair in ipairs((info.node.metadata or {}).contact_exclusions or {}) do
  if matches(pair) then allowed=true;break end
 end end
 -- A completed contact descent may wait for a cross-arm force guard before
 -- install starts. Preserve only its exact bearing intent while the same
 -- gripped body remains at that acquired endpoint. The collision test and
 -- the unchanged top/bottom-face depth gate below still apply to every pair.
 for side,receipt in pairs(K.endpoint_contacts) do
  for _,contact in ipairs(receipt.contacts) do
   if matches(contact.pair) then
    local valid=not K.active[side] and K.held[side]==receipt.part and K.grips[receipt.part]==receipt.grip
    local endpointError=0
    if valid then
     local actual=joints(side)
     for i=1,6 do
      endpointError=math.max(endpointError,math.abs(H.delta(actual[i],receipt.q[i])))
     end
     if endpointError>=(K.runtime.endpoint_tolerance_rad or .008) then valid=false end
     if K.installed[contact.lower]~=contact.slot or K.snaps[contact.lower]~=contact.snap or K.grips[contact.lower] then valid=false end
     for _,heldPart in pairs(K.held) do if heldPart==contact.lower then valid=false end end
    end
    if valid then
     if not allowed then endpointReceipts[#endpointReceipts+1]={side=side,receipt=receipt,contact=contact,error=endpointError} end
     allowed=true
    else K.endpoint_contacts[side]=nil end
   end
  end
 end
 local firstBlock,secondBlock
 for part,handle in pairs(K.blocks) do if handle==first then firstBlock=part elseif handle==second then secondBlock=part end end
 if firstBlock or secondBlock then
  local part,other,otherAlias=firstBlock or secondBlock,firstBlock and second or first,firstBlock and alias2 or alias1
  if allowed and (otherAlias:find('_Finger_') or otherAlias:find('_GripperPalm')) then
   -- Exact grasp pairs may touch a side face as well as the upper face.
   -- Bearing-surface depth tolerance below applies to table/stack/support.
   return true
  end
  for side,heldPart in pairs(K.held) do
   if heldPart==part and (otherAlias==side..'_Finger_neg' or otherAlias==side..'_Finger_pos' or otherAlias==side..'_GripperPalm') then return true end
  end
  for _,surface in ipairs((K.cfg.support_contact_surfaces or {})[part] or {}) do
   if surface==other or surface==otherAlias then allowed=true end
  end
  if other==K.cfg.table then allowed=true end
  if firstBlock and secondBlock then
   local slot=K.installed[firstBlock];local below=(K.cfg.slot_below or {})[slot] or {}
   for _,lower in ipairs(below) do if K.installed[secondBlock]==lower then allowed=true end end
   slot=K.installed[secondBlock];below=(K.cfg.slot_below or {})[slot] or {}
   for _,lower in ipairs(below) do if K.installed[firstBlock]==lower then allowed=true end end
  end
  for side,slot in pairs(K.supporting) do
   if otherAlias==side..'_SupportShoe' then
    local row=(K.cfg.slot_types or {})[slot]
    if row==K.definitions[part].type then allowed=true end
   end
  end
  if allowed then
   -- Intended body-bearing contacts are allowed only at the top/bottom faces.
   -- Deep overlap and vertical-wall intersections still fail.
   local lo1,hi1=H.worldZBounds(first);local lo2,hi2=H.worldZBounds(second)
   local gap=math.min(math.abs(lo1-hi2),math.abs(lo2-hi1))
   local accepted=gap<=(K.runtime.contact_tolerance_m or .003)
   if accepted then for _,record in ipairs(endpointReceipts) do
    if not record.contact.reported then
     record.contact.reported=true
     emit('endpoint_bearing_contact_accepted',nil,{arm=record.side,
      source_task=record.receipt.source_task,source_node=record.receipt.source_node,
      pair=clone(record.contact.pair),held=record.receipt.part,lower=record.contact.lower,
      endpoint_error_rad=record.error,face_gap_m=gap,depth_gate_verified=true,collision_measured=true})
    end
   end end
   return accepted
  end
 end
 return allowed
end
local function collision(first,second,node)
 local fixed=K.fixedInstallation[first]
 if fixed and fixed[second] then return false end
 local hit=sim.checkCollision(first,second);K.collision_checks=K.collision_checks+1
 if hit>0 and not isContact(first,second) then
  local details={first=sim.getObjectAlias(first),second=sim.getObjectAlias(second),time=sim.getSimulationTime()}
  abort('Actual collision: '..details.first..' / '..details.second,node,'collision_failure',details);return true
 end
 return false
end
local function monitorCollisions()
 local a,b=K.arms.A,K.arms.B
 local hit=sim.checkCollision(a.collision,b.collision);K.collision_checks=K.collision_checks+1
 if hit>0 then for _,one in ipairs(a.shapes) do for _,two in ipairs(b.shapes) do if collision(one,two) then return end end end end
 for _,arm in pairs(K.arms) do
  for _,pair in ipairs(arm.selfpairs) do
   local selfHit,objects=sim.checkCollision(pair[1],pair[2]);K.collision_checks=K.collision_checks+1
   if selfHit>0 and collision(objects[1],objects[2]) then return end
  end
  for _,shape in ipairs(arm.shapes) do
   if collision(shape,K.cfg.table) then return end
   for _,obstacle in ipairs(K.cfg.obstacles or {}) do if collision(shape,resolve(obstacle)) then return end end
   for _,body in pairs(K.blocks) do if collision(shape,body) then return end end
  end
 end
 for part,body in pairs(K.blocks) do
  if collision(body,K.cfg.table) then return end
  for _,obstacle in ipairs(K.cfg.obstacles or {}) do if collision(body,resolve(obstacle)) then return end end
  for other,body2 in pairs(K.blocks) do if part<other and collision(body,body2) then return end end
 end
end
function SkillnetV13.sensing()
 if not K or K.failure then return end
 K.step=K.step+1
 if K.step%5==0 then monitorCollisions();if K.failure then return end end
 local row={time=sim.getSimulationTime(),qA=joints('A'),qB=joints('B'),parts={},support={},tool={}}
 for part,body in pairs(K.blocks) do row.parts[part]=sim.getObjectPose(body,sim.handle_world) end
 for side,arm in pairs(K.arms) do
  row.support[side]=supportForce(side)
  local z,force,torque=H.sensorWorldZ(arm.tool_sensor)
  row.tool[side]={valid=z~=nil,force_N=force,torque_Nm=torque}
  row[side..'_joint_velocity_rad_s']={};row[side..'_joint_torque_Nm']={}
  for i,joint in ipairs(arm.joints) do
   local velocity=sim.getJointVelocity(joint);K.peak_velocity=math.max(K.peak_velocity,math.abs(velocity))
   row[side..'_joint_velocity_rad_s'][i]=velocity;row[side..'_joint_torque_Nm'][i]=sim.getJointForce(joint)
  end
 end
 if K.step%5==0 then K.trace[#K.trace+1]=row end
end
function SkillnetV13.dyn(data) end
function SkillnetV13.getState()
 local count=0 for _,state in pairs(K.states) do if state=='done' then count=count+1 end end
 return {done=K.done,failure=K.failure,time=sim.getSimulationTime(),completed=count,total=#K.program.nodes,
  peak_tracking_rad=K.peak_tracking,peak_velocity_rad_s=K.peak_velocity,
  active_A=K.active.A and K.active.A.node.id,active_B=K.active.B and K.active.B.node.id}
end
function SkillnetV13.getResult()
 local result=SkillnetV13.getState()
 result.events=K.events;result.trace=K.trace;result.dynamic_enablement=K.enabled
 result.node_states=K.states;result.task_schedule={};result.part_poses={};result.snap_dynamic={}
 result.joint_positions={};result.home_errors_rad={};result.support_measurements={}
 result.collision_failures=K.collision_failures;result.tracking_failures=K.tracking_failures
 result.collision_monitor={enabled=true,checks=K.collision_checks}
 result.tracking_monitor={enabled=true,samples=K.tracking_samples,limit_rad=K.runtime.tracking_limit_rad or .15,peak_error_rad=K.peak_tracking}
 result.makespan=(K.finish or sim.getSimulationTime())-K.start
 for _,task in pairs(K.times) do result.task_schedule[#result.task_schedule+1]=task end
 for part,body in pairs(K.blocks) do result.part_poses[part]=sim.getObjectPose(body,sim.handle_world) end
 for part,sensor in pairs(K.snaps) do result.snap_dynamic[part]=sim.isDynamicallyEnabled(sensor) end
 for side,arm in pairs(K.arms) do
  result.joint_positions[side]=joints(side);result.support_measurements[side]=supportForce(side)
  local error=0 for i,q in ipairs(result.joint_positions[side]) do error=math.max(error,math.abs(H.delta(q,arm.home_q[i]))) end
  result.home_errors_rad[side]=error
 end
 result.final_state={held={A=K.held.A or false,B=K.held.B or false},installed=K.installed,
  supporting=K.supporting,placed=K.placed,transfers=K.transfers}
 return result
end
function SkillnetV13.cleanup()
 if K then for _,handle in ipairs(K.collections) do sim.destroyCollection(handle) end end
end
