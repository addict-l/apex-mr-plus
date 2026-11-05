-- Completion-driven scene3 TPG dispatcher. Joint targets drive actual dynamics.
sim=require 'sim'
local cfg,program,arms,blocks,definitions,nodes,pred,states,active,held,grips,snaps
local events,trace,taskTimes,enabled,flags,runtime,done,failure,startTime,finishTime
local supportSensor,supportLift,supportTare,root,permanentSensor,initialBridgePose
local peakTracking,peakVelocity,peakSupport,stepCount,settled,pressLoad,overloads
local appliedLoadN,appliedLoadSubsteps
local assemblyGuardNodes,assemblyContactLimits
local prefix='/APEX_MR_Scene3/'

local function norm(v) local x=0 for _,a in ipairs(v) do x=x+a*a end return math.sqrt(x) end
local function finiteVector3(v)
 if type(v)~='table' or #v~=3 then return false end
 for _,x in ipairs(v) do if type(x)~='number' or x~=x or math.abs(x)==math.huge then return false end end
 return true
end
local function dist(a,b) return norm({a[1]-b[1],a[2]-b[2],a[3]-b[3]}) end
local function delta(a,b) return math.atan2(math.sin(a-b),math.cos(a-b)) end
local function orientationError(a,b)
 local dot=0 for i=4,7 do dot=dot+a[i]*b[i] end
 return 2*math.acos(math.min(1,math.abs(dot)))
end
local function resolve(alias)
 local h=sim.getObject(prefix..alias,{noError=true})
 if h~=-1 then return h end
 for _,object in ipairs(sim.getObjectsInTree(root,sim.handle_all,0)) do
  if sim.getObjectAlias(object)==alias then return object end
 end
 error('Required scene3 object not found: '..alias)
end
local function qactual(side)
 local q={} for i,j in ipairs(arms[side].joints) do q[i]=sim.getJointPosition(j) end return q
end
local function target(side,q)
 for i,j in ipairs(arms[side].joints) do sim.setJointTargetPosition(j,q[i]) end
end
local function fail(message)
 if failure then return end
 failure=message
 sim.addLog(sim.verbosity_errors,'[Scene3 APEX] '..message)
 for side,a in pairs(arms or {}) do a.hold=qactual(side) end
end
local function log(kind,node,details)
 events[#events+1]={kind=kind,time=sim.getSimulationTime(),node=node and node.id,
  task_id=node and node.task_id,arm=node and node.arm,details=details or {}}
end
local function sensorWorldZ(h)
 local status,f,t=sim.readForceSensor(h)
 if status&1==0 then return nil,nil,nil end
 local m=sim.getObjectMatrix(h,sim.handle_world)
 return m[9]*f[1]+m[10]*f[2]+m[11]*f[3],f,t
end
local function supportMeasurement()
 local z=sensorWorldZ(supportSensor)
 if not z or supportTare==nil then return 0,false end
 local force=math.abs(z-supportTare)
 peakSupport=math.max(peakSupport,force)
 return force,true
end
local function armSupportForce()
 local z=sensorWorldZ(arms.A.pad_sensor)
 if not z then return 0,false end
 return math.max(0,math.abs(z)-(runtime.A_pad_mass_kg or .03)*9.81),true
end
local function bridgeMeasurement()
 local pose=sim.getObjectPose(blocks.Bridge,sim.handle_world)
 local goal=cfg.bridge_goal_pose or initialBridgePose
 return dist(pose,goal),orientationError(pose,goal),pose
end
local function worldZBounds(shape)
 local lo={sim.getObjectFloatParam(shape,sim.objfloatparam_objbbox_min_x),
  sim.getObjectFloatParam(shape,sim.objfloatparam_objbbox_min_y),sim.getObjectFloatParam(shape,sim.objfloatparam_objbbox_min_z)}
 local hi={sim.getObjectFloatParam(shape,sim.objfloatparam_objbbox_max_x),
  sim.getObjectFloatParam(shape,sim.objfloatparam_objbbox_max_y),sim.getObjectFloatParam(shape,sim.objfloatparam_objbbox_max_z)}
 local m=sim.getObjectMatrix(shape,sim.handle_world);local lower=math.huge;local upper=-math.huge
 for _,x in ipairs({lo[1],hi[1]}) do for _,y in ipairs({lo[2],hi[2]}) do for _,z in ipairs({lo[3],hi[3]}) do
  local wz=m[9]*x+m[10]*y+m[11]*z+m[12];lower=math.min(lower,wz);upper=math.max(upper,wz)
 end end end
 return lower,upper
end
local function permanentMeasurement()
 local ep,er=bridgeMeasurement()
 local sensor=permanentSensor or snaps.K1
 if not sensor then return 0,false,ep,er,false,'none' end
 local z,f=sensorWorldZ(sensor)
 local force=f and norm(f) or 0
 local source='PermanentBridgeFT'
 if not permanentSensor then
  -- K1 may already bear the beam through actual contact before its final
  -- bridge constraint exists. Its snap transmits its own weight as well;
  -- only the measured excess vertical load qualifies as bridge bearing.
  source='K1Snap'
  force=z and math.max(0,math.abs(z)-(definitions.K1.mass or .18)*9.81) or 0
 end
 local valid=z~=nil and force>=(runtime.permanent_force_min_N or 1.) and
  ep<=(runtime.pose_tolerance_m or .009) and er<=(runtime.orientation_tolerance_rad or .04)
 return force,valid,ep,er,z~=nil,source
end
local function newSensor(name,parent,object,limit,anchor)
 local h=sim.createForceSensor(0,{0,1,5,0,0},{.006,500,limit or 10.,0,0})
 sim.setObjectAlias(h,name);sim.setObjectParent(h,parent,true)
 local pose=sim.getObjectPose(object,sim.handle_world)
 if anchor then for i=1,3 do pose[i]=anchor[i] end end
 -- Only the newly created constraint frame is positioned. Body world poses
 -- are preserved by every parent change; no object is placed by teleporting.
 sim.setObjectPose(h,pose,sim.handle_world)
 sim.setObjectInt32Param(h,sim.objintparam_visibility_layer,0)
 sim.setObjectParent(object,h,true);sim.resetDynamicObject(object)
 return h
end
local function actionOf(node)
 if node.node_type=='pose' then return nil end
 return (node.metadata or {}).action
end
local function expectedPose(node)
 local p=node.part;local meta=node.metadata or {}
 if meta.target_pose then return meta.target_pose end
 if cfg.goal_poses and cfg.goal_poses[p] then return cfg.goal_poses[p] end
 local d=definitions[p]
 if d and d.goal then
  if #d.goal==7 then return d.goal end
  return {d.goal[1],d.goal[2],d.goal[3],0,0,0,1}
 end
 if cfg.goal and cfg.goal[p] then
  local g=cfg.goal[p];return {g[1],g[2],g[3],0,0,0,1}
 end
 fail('Missing measured-placement target for '..tostring(p));return nil
end
local function poseCheck(node)
 local wanted=expectedPose(node)
 if not wanted then return nil end
 local measured=sim.getObjectPose(blocks[node.part],sim.handle_world)
 local position=dist(measured,wanted)
 local angle=#wanted>=7 and orientationError(measured,wanted) or 0
 if position>(runtime.pose_tolerance_m or .009) or angle>(runtime.orientation_tolerance_rad or .04) then
  fail(string.format('Placement %s outside tolerance: %.5fm %.4frad',node.part,position,angle));return nil
 end
 return position,angle,measured
end
local function attach(node)
 local part,side=node.part,node.arm
 if not blocks[part] then fail('Unknown grasp part '..tostring(part));return end
 if held[side] then fail('Arm already holds '..held[side]);return end
 if part=='TemporarySupport' and flags.support_placed and not flags.support_unloaded then
  fail('Cannot grasp fixture before verified unload');return
 end
 if (part=='K1' or part=='K2') and not flags.buffer[part] and not flags.installed[part] and side~='A' then
  fail('A-exclusive source supply violated for '..part);return
 end
 if flags.installed[part] then fail('Cannot pick already installed part '..part);return end
 local relative=sim.getObjectPose(blocks[part],arms[side].tip)
 local binding=(node.metadata or {}).payload_after or node.payload
 if binding and binding.offset and dist(relative,binding.offset)>(runtime.grasp_tolerance_m or .009) then
  fail('Grasp alignment outside tolerance: '..part);return
 end
 if binding and binding.quaternion then
  local wanted={0,0,0,table.unpack(binding.quaternion)}
  if orientationError(relative,wanted)>(runtime.grasp_orientation_tolerance_rad or .06) then
   fail('Grasp orientation outside tolerance: '..part);return
  end
 end
 grips[part]=newSensor('GraspFT_'..part,arms[side].palm,blocks[part],100.)
 held[side]=part
 sim.addItemToCollection(arms[side].collision,sim.handle_tree,blocks[part],0)
 if flags.buffer[part] then flags.buffer[part]=nil;flags.buffer_occupied=nil end
 log('grasp',node,{part=part,dynamic=sim.isDynamicallyEnabled(blocks[part]),relative_pose=relative})
end
local function detach(node)
 local part,side=node.part,node.arm
 if held[side]~=part or not grips[part] then fail('Release ownership mismatch '..tostring(part));return false end
 sim.addItemToCollection(arms[side].collision,sim.handle_tree,blocks[part],1)
 sim.setObjectParent(blocks[part],root,true)
 sim.removeObjects({grips[part]});grips[part]=nil
 sim.resetDynamicObject(blocks[part]);held[side]=nil
 return true
end
local function installPart(node)
 local part=node.part
 if part=='K2' and not flags.installed.K1 then fail('K2 requires installed K1');return end
 if part~='Gate' and not flags.support_verified then fail('Installation before support handoff');return end
 if part=='Gate' and not flags.permanent_verified then fail('Gate closed before permanent verification');return end
 if part=='Gate' and cfg.selected_support and cfg.selected_support.candidate=='inner_pocket' and not flags.support_extracted then
  fail('Gate would trap the inner fixture');return
 end
 local ep,er,pose=poseCheck(node);if not ep then return end
 if not detach(node) then return end
 local parent
 if part=='K1' then parent=resolve('RightTower_2_Body')
 elseif part=='K2' then parent=blocks.Bridge
 elseif part=='Gate' then parent=resolve(cfg.baseplate_alias or 'Baseplate')
 else fail('Unsupported install part '..tostring(part));return end
 local d=definitions[part] or {}
 local anchor={pose[1],pose[2],pose[3]-(d.size and d.size[3]/2 or 0)}
 snaps[part]=newSensor('SnapFT_'..part,parent,blocks[part],d.connection_moment_limit or 10.,anchor)
 flags.installed[part]=true
 log('part_installed',node,{part=part,position_error_m=ep,orientation_error_rad=er,
  dynamic=sim.isDynamicallyEnabled(blocks[part]),model='ideal rigid constraint at measured pose'})
 if part=='Gate' then flags.gate_closed=true;log('gate_closed',node,{position_error_m=ep}) end
end
local function startPermanentVerification(node)
 if not flags.installed.K1 or not flags.installed.K2 then fail('Permanent verification requires K1 and K2');return end
 if flags.arm_support_released~=true then fail('Permanent verification before arm release');return end
 if permanentSensor then fail('Permanent connection already created');return end
 local hinge=sim.getObject(prefix..'BridgeHinge',{noError=true})
 if hinge==-1 then
  for _,h in ipairs(sim.getObjectsInTree(root,sim.object_joint_type,0)) do
   if sim.getObjectAlias(h)=='BridgeHinge' then hinge=h;break end
  end
 end
 local bridgePose=sim.getObjectPose(blocks.Bridge,sim.handle_world)
 sim.setObjectParent(blocks.Bridge,root,true)
 if hinge~=-1 then sim.removeObjects({hinge}) end
 local anchor=sim.getObjectPosition(blocks.K1,sim.handle_world)
 local size=(definitions.K1 or {}).size
 anchor[3]=anchor[3]+(size and size[3]/2 or .04)
 permanentSensor=newSensor('PermanentBridgeFT',blocks.K1,blocks.Bridge,
  runtime.permanent_moment_limit_Nm or 10.,anchor)
 snaps.Bridge=permanentSensor
 flags.permanent_verifying=true
 -- Proof loading occurs with the temporary shoe lowered. This is the load
 -- transfer test inside verification, not permission to extract the fixture.
 sim.setJointTargetPosition(supportLift,0.)
 log('permanent_proof_test_started',node,{bridge_pose_before=bridgePose,
  load_N=runtime.proof_load_N or 8.,target_retraction_m=runtime.support_extension_m or .008})
end
local function beginNode(node,t)
 local action=actionOf(node)
 if node.arm=='A' and not flags.support_verified then
  local idle=(action=='wait' or action=='hold') and #(node.motion_segments or {})==0
  if not idle then fail('A motion before measured fixture handoff');return end
 end
 if node.arm=='A' and flags.support_verified and not flags.arm_support_released then
  local idle=(action=='wait' or action=='hold') and #(node.motion_segments or {})==0
  if action~='release_arm_support' and not idle then fail('A task before explicit arm-support release');return end
 end
 if action=='release_arm_support' and not flags.support_verified then fail('Premature arm release');return end
 if action=='verify_support' then
  if not flags.support_placed or held.A=='TemporarySupport' or held.B=='TemporarySupport' then
   fail('Support load check requires released fixture on table');return
  end
  sim.setJointTargetPosition(supportLift,runtime.support_extension_m or .008)
 elseif action=='verify_permanent' then
  startPermanentVerification(node);if failure then return end
 elseif action=='unload_support' then
  if not flags.permanent_verified then fail('Support unload before permanent proof test');return end
  sim.setJointTargetPosition(supportLift,0.)
 elseif action=='extract_support' then
  if not flags.support_unloaded then fail('Extraction before measured unload');return end
 end
 states[node.id]='running';active[node.arm]={node=node,start=t,within_since=nil}
 local key=node.task_id or node.id
 if not taskTimes[key] then taskTimes[key]={id=key,task_id=key,arm=node.arm,kind=node.kind,part=node.part,start=t} end
 log('node_started',node)
end
local function plannedQ(node,elapsed)
 local remaining=elapsed
 for _,segment in ipairs(node.motion_segments or {}) do
  if remaining<=segment.duration then
   local u=math.max(0,math.min(1,remaining/segment.duration));local s=u*u*u*(10+u*(-15+6*u))
   local q={} for i=1,6 do q[i]=segment.q0[i]+(segment.q1[i]-segment.q0[i])*s end return q
  end
  remaining=remaining-segment.duration
 end
 return {table.unpack(node.q_end or node.q_samples[#node.q_samples])}
end
local function actionQualified(node)
 local action=actionOf(node)
 local force,valid=supportMeasurement()
 if action=='verify_support' then return valid and force>=(runtime.support_force_min_N or 1.) end
 if action=='verify_permanent' then
  local _,permanentValid=permanentMeasurement()
  return permanentValid and valid and force<=(runtime.unload_force_max_N or .5) and
   math.abs(sim.getJointPosition(supportLift))<=(runtime.lift_tolerance_m or .0001)
 end
 if action=='unload_support' then
  return flags.permanent_verified and valid and force<=(runtime.unload_force_max_N or .5) and
   math.abs(sim.getJointPosition(supportLift))<=(runtime.lift_tolerance_m or .0001)
 end
 return true
end
local function finishNode(info,t)
 local node=info.node;local action=actionOf(node)
 if action=='grasp' then attach(node)
 elseif action=='release_buffer' then
  if flags.buffer_occupied then fail('Shared buffer capacity exceeded');return end
  local ep=poseCheck(node);if not ep then return end
  if not detach(node) then return end
  flags.buffer[node.part]=true;flags.buffer_occupied=node.part
  log('buffer_deposit',node,{part=node.part,position_error_m=ep})
 elseif action=='release_support' then
  local ep=poseCheck(node);if not ep then return end
  if not detach(node) then return end
  flags.support_placed=true
  flags.support_placed_pose=sim.getObjectPose(blocks.TemporarySupport,sim.handle_world)
  log('support_placed',node,{position_error_m=ep,table_contact_model='dynamic gravity contact; no fixed constraint'})
 elseif action=='verify_support' then
  local force,valid=supportMeasurement()
  flags.support_verified=true
  log('support_load_verified',node,{valid=valid,force_N=force,tare_world_z_N=supportTare})
 elseif action=='release_arm_support' then
  if not flags.support_verified then fail('Arm release lacks load verification');return end
  flags.arm_support_released=true
  log('arm_support_released',node,{support_force_N=supportMeasurement(),pad_remains_respondable=true})
 elseif action=='deactivate_arm_pad' then
  local force,valid=supportMeasurement()
  if not flags.arm_support_released or not flags.support_verified or not valid or
    force<(runtime.continuity_force_min_N or .15) then fail('Cannot deactivate A pad before measured support handoff');return end
  local bridgeBottom=worldZBounds(blocks.Bridge)
  local _,padTop=worldZBounds(arms.A.pad)
  local clearance=bridgeBottom-padTop
  if clearance<.005 then fail('A pad must physically clear bridge by 5mm before deactivation');return end
  sim.setObjectInt32Param(arms.A.pad,sim.shapeintparam_respondable,0)
  sim.resetDynamicObject(arms.A.pad)
  flags.A_pad_deactivated=true
  log('pad_deactivated',node,{arm='A',measured_vertical_clearance_m=clearance,
   support_force_N=force,respondable=false,mass_and_planning_geometry_retained=true})
 elseif action=='install' or action=='close_gate' then installPart(node)
 elseif action=='verify_permanent' then
  local force,valid,ep,er=permanentMeasurement()
  flags.permanent_verified=true;flags.permanent_verifying=false
  log('permanent_structure_verified',node,{valid=valid,force_N=force,
   proof_load_N=runtime.proof_load_N or 8.,position_error_m=ep,orientation_error_rad=er,
   temporary_support_force_N=supportMeasurement(),shoe_position_m=sim.getJointPosition(supportLift),
   independent_load_test=true})
 elseif action=='unload_support' then
  flags.support_unloaded=true
  log('support_unloaded',node,{retraction_m=(runtime.support_extension_m or .008)-sim.getJointPosition(supportLift),
   force_N=supportMeasurement(),permanent_verified=flags.permanent_verified})
 elseif action=='extract_support' then
  if not flags.support_unloaded or held[node.arm]~='TemporarySupport' then fail('Invalid support extraction ownership');return end
  if cfg.selected_support and cfg.selected_support.candidate=='inner_pocket' and flags.gate_closed then
   fail('Inner support trapped by closed gate');return
  end
  local pose=sim.getObjectPose(blocks.TemporarySupport,sim.handle_world)
  local displacement=flags.support_placed_pose and dist(pose,flags.support_placed_pose) or 0
  if displacement<(runtime.minimum_extraction_displacement_m or .10) then fail('Fixture has not cleared its support position');return end
  flags.support_extracted=true;log('support_extracted',node,{pose=pose,displacement_m=displacement})
 elseif action=='return_support' then
  if not flags.support_extracted then fail('Support returned before extraction');return end
  local ep=poseCheck(node);if not ep then return end
  if not detach(node) then return end
  flags.support_returned=true;log('support_returned',node,{position_error_m=ep})
 elseif action~=nil and action~='wait' and action~='hold' then
  fail('Unknown skill action '..tostring(action));return
 end
 if failure then return end
 states[node.id]='done';active[node.arm]=nil;arms[node.arm].hold={table.unpack(node.q_end)}
 local task=taskTimes[node.task_id or node.id];task['end']=t;task.duration=t-task.start
 log('node_completed',node)
end

function sysCall_init()
 root=sim.getObject('/APEX_MR_Scene3');cfg=sim.readCustomTableData(root,'Scene3APEX')
 runtime=cfg.runtime or {};program=cfg.program
 assemblyContactLimits={force_N=runtime.assembly_tool_force_limit_N or 40.,
  torque_Nm=runtime.assembly_tool_torque_limit_Nm or 8.,basis='experiment limits; not robot ratings'}
 assert(assemblyContactLimits.force_N>0 and assemblyContactLimits.force_N<math.huge and
  assemblyContactLimits.torque_Nm>0 and assemblyContactLimits.torque_Nm<math.huge,'Invalid assembly contact limits')
 assemblyGuardNodes={}
 assert(program and program.nodes and #program.nodes>0,'No scene3 TPG embedded')
 arms={};blocks={};definitions={};nodes={};pred={};states={};active={};held={};grips={};snaps={}
 events={};trace={};taskTimes={};enabled={};flags={buffer={},installed={}};overloads={}
 done=false;failure=nil;settled=false;startTime=sim.getSimulationTime();stepCount=0
 supportTare=nil;permanentSensor=nil;finishTime=nil;pressLoad=nil
 appliedLoadN=0;appliedLoadSubsteps=0
 peakTracking=0;peakVelocity=0;peakSupport=0
 for _,part in ipairs(cfg.parts) do
  blocks[part.id]=resolve(part.alias or part.id);definitions[part.id]=part
 end
 for _,name in ipairs({'K1','K2','Gate','TemporarySupport','Bridge'}) do blocks[name]=blocks[name] or resolve(name) end
 supportSensor=resolve('SupportLoadSensor');supportLift=resolve('SupportLift')
 initialBridgePose=sim.getObjectPose(blocks.Bridge,sim.handle_world)
 sim.setJointTargetPosition(supportLift,0.)
 for _,side in ipairs({'A','B'}) do
  local model=resolve(side..'_UR5');local a={model=model,joints={}}
  for i=1,6 do a.joints[i]=sim.getObject(':/'..side..'_joint_'..i,{proxy=model}) end
  a.tip=sim.getObject(':/'..side..'_TCP',{proxy=model})
  a.palm=sim.getObject(':/'..side..'_GripperPalm',{proxy=model})
  a.pad=sim.getObject(':/'..side..'_SupportShoe',{proxy=model})
  a.pad_sensor=sim.getObject(':/'..side..'_PadForceSensor',{proxy=model})
  a.tool_sensor=sim.getObject(':/'..side..'_ToolForceSensor',{proxy=model})
  a.collision=sim.createCollection(1)
  for _,h in ipairs(sim.getObjectsInTree(model,sim.object_shape_type,0)) do
   local alias=sim.getObjectAlias(h)
   local physical=h==model or sim.getObjectInt32Param(h,sim.shapeintparam_static)==0 or
    sim.getObjectInt32Param(h,sim.shapeintparam_respondable)~=0 or string.match(alias,'_Finger')
   if physical then sim.addItemToCollection(a.collision,sim.handle_single,h,0) end
  end
  arms[side]=a;a.hold=qactual(side)
 end
 for _,node in ipairs(program.nodes) do
  nodes[node.id]=node;pred[node.id]={};states[node.id]='pending';node.metadata=node.metadata or {}
 end
 for _,edge in ipairs(program.edges) do pred[edge.target][#pred[edge.target]+1]=edge.source end
end

function sysCall_actuation()
 local t=sim.getSimulationTime();pressLoad=nil
 appliedLoadN=0;appliedLoadSubsteps=0
 assemblyGuardNodes={}
 if failure or done then for side,a in pairs(arms) do target(side,a.hold) end return end
 if t-startTime<.6 then for side,a in pairs(arms) do target(side,a.hold) end return end
 if not settled then
  if supportTare==nil then supportTare=sensorWorldZ(supportSensor) end
  local force,valid=armSupportForce()
  if not valid or force<(runtime.initial_arm_force_min_N or .15) or supportTare==nil then
   for side,a in pairs(arms) do target(side,a.hold) end
   if t-startTime>3. then fail('Initial A support or unloaded fixture sensor did not become measurable') end
   return
  end
  settled=true
  for side,a in pairs(arms) do
   enabled[side]={joints={},tool_sensor=sim.isDynamicallyEnabled(a.tool_sensor),pad_sensor=sim.isDynamicallyEnabled(a.pad_sensor),pad=sim.isDynamicallyEnabled(a.pad)}
   for i,j in ipairs(a.joints) do
    enabled[side].joints[i]=sim.isDynamicallyEnabled(j)
    if not enabled[side].joints[i] then fail('Dynamic joint disabled '..side..i) end
   end
  end
  enabled.support_sensor=sim.isDynamicallyEnabled(supportSensor)
  enabled.support_lift=sim.isDynamicallyEnabled(supportLift)
  enabled.objects={};for part,h in pairs(blocks) do enabled.objects[part]=sim.isDynamicallyEnabled(h) end
  if not enabled.support_sensor or not enabled.support_lift then fail('Support actuator or load sensor dynamics disabled') end
  log('initial_arm_support_verified',nil,{force_N=force,fixture_tare_world_z_N=supportTare})
  log('execution_ready',nil,{dynamic_enablement=enabled,assembly_contact_limits=assemblyContactLimits})
 end
 for _,node in ipairs(program.nodes) do
  if states[node.id]=='pending' and not active[node.arm] then
   local ready=true for _,previous in ipairs(pred[node.id]) do if states[previous]~='done' then ready=false;break end end
   if ready then beginNode(node,t);if failure then return end end
  end
 end
 for _,side in ipairs({'A','B'}) do
  local info=active[side]
  if info then
   local node=info.node;local elapsed=t-info.start;local q=plannedQ(node,elapsed)
   -- Keep the guard for this entire physics step, including a node which
   -- completes below and consequently clears active before sensing runs.
   if node.metadata.assembly_force_guard then assemblyGuardNodes[side]=node end
   target(side,q)
   local actual=qactual(side);local error=0
   for i=1,6 do error=math.max(error,math.abs(delta(actual[i],q[i]))) end
   peakTracking=math.max(peakTracking,error)
   if error>(runtime.tracking_limit_rad or .15) and elapsed>.5 then fail('Servo tracking limit on '..side) end
   local action=actionOf(node)
   local load=action and ((node.metadata or {}).force_N or 0) or 0
   if action=='verify_permanent' then load=runtime.proof_load_N or 8. end
   if load>0 then
    if not flags.support_verified then fail('Insertion/proof load before support handoff')
    else pressLoad={force=load,position=node.metadata.load_position or { .24,.20,1.052 }} end
   end
   if elapsed>=node.duration then
    local qualified=error<(runtime.endpoint_tolerance_rad or .008) and actionQualified(node)
    if qualified then
     info.within_since=info.within_since or t
     local dwell=node.node_type=='skill' and (node.metadata.dwell_s or .20) or .04
     if action=='verify_support' then dwell=math.max(dwell,runtime.loaded_dwell_s or .30) end
     if action=='verify_permanent' then dwell=math.max(dwell,runtime.permanent_dwell_s or .65) end
     if t-info.within_since>=dwell then finishNode(info,t) end
    else info.within_since=nil end
   end
   if elapsed>node.duration+(runtime.endpoint_timeout_s or 10.) then fail('Endpoint or physical guard did not settle: '..node.id) end
  else target(side,arms[side].hold) end
 end
 local complete=true for _,state in pairs(states) do if state~='done' then complete=false;break end end
 if complete and not failure then
  if not flags.support_returned or not flags.gate_closed or not flags.permanent_verified or held.A or held.B then
   fail('TPG finished without lifecycle goal state');return
  end
  done=true;finishTime=t;log('assembly_completed',nil)
 end
end

function sysCall_dyn(inData)
 if not inData.afterStep and pressLoad and not failure then
  local h=blocks.Bridge;local p=sim.getObjectPosition(h,sim.handle_world)
  local x,y=pressLoad.position[1]-p[1],pressLoad.position[2]-p[2]
  sim.addForceAndTorque(h|sim.handleflag_resetforcetorque,{0,0,-pressLoad.force},{-y*pressLoad.force,x*pressLoad.force,0})
  -- Record only after the physics force call succeeds. The command remains
  -- separately visible as insertion_load_N; a missing dynamics callback must
  -- never be mistaken for a successfully applied proof-load substep.
  appliedLoadN=pressLoad.force;appliedLoadSubsteps=appliedLoadSubsteps+1
 end
end

function sysCall_sensing()
 stepCount=stepCount+1
 if not arms or not arms.A then return end
 local toolMeasurements={};local assemblyActive=false
 for side,a in pairs(arms) do
  local z,f,torque=sensorWorldZ(a.tool_sensor)
  local valid=z~=nil and finiteVector3(f) and finiteVector3(torque)
  toolMeasurements[side]={valid=valid,force=f,torque=torque}
  local node=assemblyGuardNodes[side]
  if node then
   assemblyActive=true
   local forceNorm=valid and norm(f) or nil
   local torqueNorm=valid and norm(torque) or nil
   if not failure and (not valid or forceNorm>assemblyContactLimits.force_N or torqueNorm>assemblyContactLimits.torque_Nm) then
    log('assembly_contact_guard_failed',node,{sensor_valid=valid,force_N=valid and f or nil,
     torque_Nm=valid and torque or nil,force_norm_N=forceNorm,torque_norm_Nm=torqueNorm,
     limits=assemblyContactLimits})
    fail(valid and string.format('Assembly tool load limit on %s: %.3f N / %.3f Nm',side,forceNorm,torqueNorm)
     or ('Assembly tool force sensor invalid on '..side))
   end
  end
 end
 for _,a in pairs(arms) do
  for _,j in ipairs(a.joints) do peakVelocity=math.max(peakVelocity,math.abs(sim.getJointVelocity(j))) end
 end
 local supportForce,supportValid=supportMeasurement()
 local permanentForce,permanentValid,bridgeError,bridgeAngle,permanentRawValid,bearingSensor=permanentMeasurement()
 if settled and flags.arm_support_released and not flags.permanent_verified and not failure then
  local Aforce,Avalid=armSupportForce()
  local continuous=(supportValid and supportForce>=(runtime.continuity_force_min_N or .15)) or
    permanentValid or (Avalid and Aforce>=(runtime.initial_arm_force_min_N or .15))
  overloads.support_loss=continuous and 0 or (overloads.support_loss or 0)+1
  if overloads.support_loss>=10 then fail('Measured support continuity lost') end
 end
 if settled and bridgeError>(runtime.bridge_failure_displacement_m or .02) then fail('Bridge moved outside support envelope') end
 if stepCount%5==0 then
  local hit,pair=sim.checkCollision(arms.A.collision,arms.B.collision)
  if hit>0 then fail('Actual inter-arm collision: '..sim.getObjectAlias(pair[1],1)..' / '..sim.getObjectAlias(pair[2],1)) end
 end
 for part,h in pairs(snaps or {}) do
  local status,force,torque=sim.readForceSensor(h)
  if status&1~=0 then
   local limit=(definitions[part] or {}).connection_moment_limit or (runtime.permanent_moment_limit_Nm or 10.)
   overloads[part]=norm(torque)>limit and (overloads[part] or 0)+1 or 0
   if overloads[part]>=10 then fail('Measured connection torque limit at '..part) end
  end
 end
 -- Preserve every guarded measurement, including the first violating step;
 -- unguarded motion retains the usual decimated trace cadence.
 if stepCount%5~=0 and not assemblyActive then return end
 local Aforce,Avalid=armSupportForce()
 local row={time=sim.getSimulationTime(),qA=qactual('A'),qB=qactual('B'),parts={},snap={},
  support_force_N=supportForce,support_valid=supportValid,A_support_force_N=Aforce,A_support_valid=Avalid,
  permanent_force_N=permanentForce,permanent_support_valid=permanentValid,
  permanent_sensor_valid=permanentRawValid,
  bearing_sensor=bearingSensor,
  bridge_position_error_m=bridgeError,bridge_orientation_error_rad=bridgeAngle,
  support_lift_position_m=sim.getJointPosition(supportLift),
  insertion_load_N=pressLoad and pressLoad.force or 0,
  applied_load_N=appliedLoadN,applied_load_substeps=appliedLoadSubsteps,
  permanent_verification_active=flags.permanent_verifying or false}
 for part,h in pairs(blocks) do row.parts[part]=sim.getObjectPose(h,sim.handle_world) end
 for part,h in pairs(snaps) do
  local status,force,torque=sim.readForceSensor(h);row.snap[part]={valid=status&1~=0}
  if status&1~=0 then row.snap[part].force_N=force;row.snap[part].torque_Nm=torque;row.snap[part].torque_norm_Nm=norm(torque) end
 end
 for side,a in pairs(arms) do
  local measurement=toolMeasurements[side]
  row[side..'_tool_sensor_valid']=measurement.valid
  row[side..'_assembly_contact']=assemblyGuardNodes[side]~=nil
  if measurement.valid then
   row[side..'_tool_force_N']=measurement.force;row[side..'_tool_torque_Nm']=measurement.torque
  end
  row[side..'_joint_torque_Nm']={};row[side..'_joint_velocity_rad_s']={}
  for i,j in ipairs(a.joints) do
   row[side..'_joint_torque_Nm'][i]=sim.getJointForce(j)
   row[side..'_joint_velocity_rad_s'][i]=sim.getJointVelocity(j)
  end
 end
 trace[#trace+1]=row
end

function getState()
 local count=0 for _,state in pairs(states or {}) do if state=='done' then count=count+1 end end
 local force,valid=supportMeasurement()
 return {done=done,failure=failure,time=sim.getSimulationTime(),completed=count,total=#program.nodes,
  peak_tracking_rad=peakTracking,peak_velocity_rad_s=peakVelocity,peak_support_normal_N=peakSupport,
  support_force_N=force,support_valid=valid,permanent_verified=flags.permanent_verified or false,
  support_load_verified=flags.support_verified or false,
  support_unloaded=flags.support_unloaded or false,
  active_A=active.A and active.A.node.id,active_B=active.B and active.B.node.id}
end
function getResult()
 local result=getState();result.events=events;result.trace=trace;result.dynamic_enablement=enabled
 result.assembly_contact_limits=assemblyContactLimits
 result.node_states=states;result.task_schedule={};result.part_poses={};result.makespan=(finishTime or sim.getSimulationTime())-startTime
 for _,task in pairs(taskTimes) do result.task_schedule[#result.task_schedule+1]=task end
 for part,h in pairs(blocks) do result.part_poses[part]=sim.getObjectPose(h,sim.handle_world) end
 result.snap_dynamic={};for part,h in pairs(snaps) do result.snap_dynamic[part]=sim.isDynamicallyEnabled(h) end
 result.final_state={held={A=held.A or false,B=held.B or false},support_returned=flags.support_returned or false,
  gate_closed=flags.gate_closed or false,permanent_verified=flags.permanent_verified or false,
  installed=flags.installed,support_unloaded=flags.support_unloaded or false}
 return result
end
function sysCall_cleanup()
 for _,a in pairs(arms or {}) do if a.collision then sim.destroyCollection(a.collision) end end
end
