-- Adapted from the validated DualArmBlocks simulator backend.
-- A/B arms; real simIK and simOMPL RRTConnect; snapshot/restore on every query.
-- Genuine OMPL planning, exact simulator collision geometry, transactional queries.
-- Kinematic planning, not force/contact dynamics. Adjacent rigid links are the
-- only automatic self-collision exclusions; grasp contacts must be explicit.
sim = require 'sim'
simIK = require 'simIK'
simOMPL = require 'simOMPL'
local C,A,collections,tasks,context={},{},{},{},nil
local collectionSet={}
local checks,ikCalls,lastProblem=0,0,nil
local COLLISION_POLICY='native base + dynamic/respondable collision shapes + explicit cuboid Palm/Finger/SupportShoe tools; purely visual static shells excluded'

function sysCall_init() end
local function copy(a) local r={} for k,v in pairs(a) do r[k]=v end return r end
local function getq(arm) local q={} for _,j in ipairs(arm.joints) do q[#q+1]=sim.getJointPosition(j) end return q end
local function setq(arm,q) for i,j in ipairs(arm.joints) do sim.setJointPosition(j,q[i]) end end
local function dist(a,b) local d=0 for i=1,#a do d=d+(a[i]-b[i])^2 end return math.sqrt(d) end
local function wrappedDist(a,b)
    local d=0 for i=1,#a do local x=math.atan(math.sin(a[i]-b[i]),math.cos(a[i]-b[i])) d=d+x*x end return math.sqrt(d)
end
local function clock() return sim.getSystemTime() end
local function resolve(h)
    if type(h)=='number' then return h end
    if C.blocks and C.blocks[h] then return C.blocks[h] end
    if h:sub(1,1)=='/' then return sim.getObject(h) end
    if C.aliases and C.aliases[h] then return C.aliases[h] end
    return sim.getObject('/APEX_MR_Plus/'..h)
end
local function pose(h) return sim.getObjectPose(h,sim.handle_world) end
local function collection(shapes)
    local c=sim.createCollection(1)
    for _,h in ipairs(shapes) do sim.addItemToCollection(c,sim.handle_single,h,0) end
    collections[#collections+1]=c
    collectionSet[c]=true
    return c
end
local function snapshot()
    local s={arms={},blocks={}}
    for side,arm in pairs(A) do s.arms[side]={q=getq(arm),target=pose(arm.target)} end
    for name,h in pairs(C.blocks or {}) do s.blocks[name]=pose(h) end
    return s
end
local function restore(s)
    for side,t in pairs(s.arms) do setq(A[side],t.q) sim.setObjectPose(A[side].target,t.target,sim.handle_world) end
    for name,p in pairs(s.blocks) do sim.setObjectPose(C.blocks[name],p,sim.handle_world) end
end
local function transaction(f)
    local s=snapshot()
    local ok,r=xpcall(f,debug.traceback)
    local restored,err=pcall(restore,s)
    if not restored then return {ok=false,error='Transaction restore failed: '..tostring(err)} end
    if not ok then return {ok=false,error=tostring(r)} end
    return r
end
local function applyPlaced(placed)
    for name,p in pairs(placed or {}) do
        local h=resolve(name)
        if #p==3 then sim.setObjectPosition(h,sim.handle_world,p) else sim.setObjectPose(h,p,sim.handle_world) end
    end
end
local function payloadSpec(side,p)
    if type(p)=='string' then p={name=p} end
    if p and p.part and not p.name then p=copy(p) p.name=p.part end
    if not p or not p.name then return nil end
    local s=copy(p)
    s.handle=resolve(p.name)
    if not p.offset then
        local rel=sim.multiplyPoses(sim.getPoseInverse(pose(A[side].tip)),pose(s.handle))
        s.offset={rel[1],rel[2],rel[3]}
        s.quaternion={rel[4],rel[5],rel[6],rel[7]}
    end
    s.quaternion=s.quaternion or {0,0,0,1}
    return s
end
local function movePayload(side,p)
    if p then
        sim.setObjectPose(p.handle,sim.multiplyPoses(pose(A[side].tip),
            {p.offset[1],p.offset[2],p.offset[3],table.unpack(p.quaternion)}),sim.handle_world)
    end
end
local function zExtent(h)
    local mn,mx={},{}
    for i,axis in ipairs({'x','y','z'}) do
        mn[i]=sim.getObjectFloatParam(h,sim['objfloatparam_objbbox_min_'..axis])
        mx[i]=sim.getObjectFloatParam(h,sim['objfloatparam_objbbox_max_'..axis])
    end
    local m=sim.getObjectMatrix(h,sim.handle_world)
    local z=m[12] local half=0
    for i=1,3 do z=z+m[8+i]*(mn[i]+mx[i])/2 half=half+math.abs(m[8+i])*(mx[i]-mn[i])/2 end
    return z-half,z+half
end
local function supportContact(a,b,ctx)
    local blockA=C.blockHandles[a]
    local blockB=C.blockHandles[b]
    local surfaceA=C.surfaceHandles and C.surfaceHandles[a]
    local surfaceB=C.surfaceHandles and C.surfaceHandles[b]
    if (blockA and (blockB or surfaceB)) or (blockB and surfaceA) then
        local bottomA,topA=zExtent(a)
        local bottomB,topB=zExtent(b)
        local tolerance=ctx.options.contact_tolerance or .002
        if surfaceA then return math.abs(bottomB-topA)<=tolerance end
        if surfaceB then return math.abs(bottomA-topB)<=tolerance end
        return math.abs(bottomA-topB)<=tolerance or math.abs(bottomB-topA)<=tolerance
    end
    return true
end
local function excluded(h1,h2,ctx)
    for _,e in ipairs(ctx.options.contact_exclusions or {}) do
        local a,b=resolve(e[1]),resolve(e[2])
        if (a==h1 and b==h2) or (a==h2 and b==h1) then return supportContact(h1,h2,ctx) end
    end
    return false
end
local function hit(x,y,label,ctx)
    local result,pair=sim.checkCollision(x,y)
    if result>0 then
        if pair and excluded(pair[1],pair[2],ctx) then
            -- Explicit pair exclusions must not suppress another collision in
            -- the same collection: test the remaining actual shape pairs.
            local xs=collectionSet[x] and sim.getCollectionObjects(x) or {x}
            local ys=collectionSet[y] and sim.getCollectionObjects(y) or {y}
            for _,a in ipairs(xs) do for _,b in ipairs(ys) do
                if not excluded(a,b,ctx) and sim.checkCollision(a,b)>0 then
                    lastProblem={reason=label,pair={sim.getObjectAlias(a,1),sim.getObjectAlias(b,1)}} return true
                end
            end end
            return false
        end
        lastProblem={reason=label,pair={}}
        if pair then for _,h in ipairs(pair) do lastProblem.pair[#lastProblem.pair+1]=sim.getObjectAlias(h,1) end end
        return true
    end
    local clearance=ctx.options.obstacle_clearance or 0
    if clearance>0 and label:find('obstacle') then
        local near=sim.checkDistance(x,y,clearance)
        if near>0 then
            lastProblem={reason=label..':clearance',required_m=clearance}
            return true
        end
    end
    return false
end
local function quaternionError(a,b)
    local dot=0 for i=1,4 do dot=dot+a[i]*b[i] end
    return 2*math.acos(math.min(1,math.abs(dot)))
end
local function specPose(spec)
    return {spec.offset[1],spec.offset[2],spec.offset[3],table.unpack(spec.quaternion or {0,0,0,1})}
end
local function desiredTCP(center,ctx,side)
    local grasps=ctx.grasps or ctx.options.grasp_transforms
    if grasps and grasps[side] then
        local objectPose={center[1],center[2],center[3],table.unpack(ctx.object_quaternion or {0,0,0,1})}
        return sim.multiplyPoses(objectPose,sim.getPoseInverse(specPose(grasps[side])))
    end
    local offset=ctx.offset or ctx.options.offset or .19
    local height=ctx.grasp_z or ctx.options.grasp_z or ctx.options.tcp_height or .028
    local quats=ctx.quaternions or ctx.options.quaternions or {A=A.A.quaternion,B=A.B.quaternion}
    return {center[1]+(side=='A' and -offset or offset),center[2],center[3]+height,table.unpack(quats[side])}
end
local function closedChain(ctx)
    if not ctx.shared_payload then return true end
    local obj=pose(ctx.shared_payload)
    for _,side in ipairs({'A','B'}) do
        local actual=pose(A[side].tip)
        local desired
        if ctx.grasps or ctx.options.grasp_transforms then
            local grasps=ctx.grasps or ctx.options.grasp_transforms
            desired=sim.multiplyPoses(obj,sim.getPoseInverse(specPose(grasps[side])))
        else desired=desiredTCP({obj[1],obj[2],obj[3]},ctx,side) end
        local positionError=dist({actual[1],actual[2],actual[3]},{desired[1],desired[2],desired[3]})
        local orientationError=quaternionError({actual[4],actual[5],actual[6],actual[7]},
            {desired[4],desired[5],desired[6],desired[7]})
        if positionError>(ctx.options.closure_tolerance or .002) or orientationError>(ctx.options.orientation_tolerance or .01) then
            lastProblem={reason='closed_chain_object_frame:'..side,error_m=positionError,error_rad=orientationError}
            return false
        end
    end
    return true
end
local function geometry(ctx)
    checks=checks+1
    if not closedChain(ctx) then return false end
    for _,side in ipairs({'A','B'}) do
        local arm=A[side]
        for _,e in ipairs(arm.selfpairs) do if hit(e[1],e[2],'self_collision:'..side,ctx) then return false end end
        if hit(arm.moving,C.table,'table_collision:'..side,ctx) then return false end
        for _,h in ipairs(C.obstacles) do if hit(arm.all,h,'obstacle_collision:'..side,ctx) then return false end end
    end
    if hit(A.A.all,A.B.all,'interarm_collision',ctx) then return false end
    local payloadHandles={}
    for side,p in pairs(ctx.payloads) do if p then payloadHandles[p.handle]=side end end
    for name,h in pairs(C.blocks) do
        local holder=payloadHandles[h]
        for side,arm in pairs(A) do
            -- Contact between an attached block and the terminal tool is
            -- allowed. Every earlier link and the entire opposite arm is checked.
            local coll=(holder==side) and arm.withoutTool or arm.all
            if not (ctx.shared_payload==h and side=='B') and hit(coll,h,'block_robot_collision:'..name..':'..side,ctx) then return false end
            if ctx.shared_payload==h and side=='B' and hit(arm.withoutTool,h,'shared_payload_robot_collision',ctx) then return false end
        end
        if holder then
            if hit(h,C.table,'payload_table_collision:'..name,ctx) then return false end
            for _,o in ipairs(C.obstacles) do if hit(h,o,'payload_obstacle_collision:'..name,ctx) then return false end end
            for other,k in pairs(C.blocks) do if other~=name and hit(h,k,'payload_block_collision:'..name..':'..other,ctx) then return false end end
        end
    end
    return true
end
local function newContext(payloads,placed,options,activeSide)
    applyPlaced(placed)
    local ctx={payloads={},options=options or {}}
    if activeSide then
        local other=activeSide=='A' and 'B' or 'A'
        if ctx.options.other_q then setq(A[other],ctx.options.other_q) end
        if ctx.options.other_payload then ctx.payloads[other]=payloadSpec(other,ctx.options.other_payload) end
    end
    for side,p in pairs(payloads or {}) do ctx.payloads[side]=payloadSpec(side,p) end
    if ctx.options.shared_payload then ctx.shared_payload=resolve(ctx.options.shared_payload) end
    return ctx
end
local function validJoint(side,q,ctx)
    setq(A[side],q)
    for owner,p in pairs(ctx.payloads) do movePayload(owner,p) end
    return geometry(ctx)
end
local function ikFromSeed(side,p,quat,seed)
    local arm=A[side]
    setq(arm,seed)
    sim.setObjectPose(arm.target,{p[1],p[2],p[3],table.unpack(quat)},sim.handle_world)
    simIK.syncFromSim(arm.env,{arm.group})
    ikCalls=ikCalls+1
    local r=simIK.handleGroup(arm.env,arm.group,{syncWorlds=true,allowError=false})
    if r~=simIK.result_success or dist(sim.getObjectPosition(arm.tip,sim.handle_world),p)>.001 then return nil end
    local q=getq(arm)
    -- Preserve joint winding. simIK can report an equivalent revolution across
    -- +/-pi; that is a coordinate wrap, not a physical branch transition.
    for i,j in ipairs(arm.joints) do
        local cyclic,interval=sim.getJointInterval(j)
        local low,high=interval[1],interval[1]+interval[2]
        if cyclic then low=-2*math.pi high=2*math.pi end
        while q[i]-seed[i]>math.pi and q[i]-2*math.pi>=low do q[i]=q[i]-2*math.pi end
        while q[i]-seed[i]<-math.pi and q[i]+2*math.pi<=high do q[i]=q[i]+2*math.pi end
    end
    setq(arm,q)
    return q
end
local function serialize(v)
    if type(v)~='table' then return tostring(v) end
    local keys={} for k in pairs(v) do keys[#keys+1]=k end table.sort(keys,function(a,b)return tostring(a)<tostring(b)end)
    local out={'{'} for _,k in ipairs(keys) do out[#out+1]=tostring(k)..'='..serialize(v[k])..';' end out[#out+1]='}' return table.concat(out)
end
local function meshFingerprint(h)
    local vertices,indices=sim.getShapeMesh(h)
    local first,second=17,31
    for _,x in ipairs(vertices) do
        local value=math.floor(x*1e7+.5)
        first=(first*131+value)%2147483647
        second=(second*137+value)%2147483629
    end
    for _,i in ipairs(indices) do first=(first*131+i)%2147483647 second=(second*137+i)%2147483629 end
    return {#vertices,#indices,first,second}
end
local function signature(side,ctx,algorithm)
    local geometryOptions=copy(ctx.options)
    for _,k in ipairs({'seed','seeds','start_q','ik_time','solve_time','simplify_time','max_joint_step','multiple','max_dist','center_step','max_branch_step'}) do geometryOptions[k]=nil end
    local world={side=side,algorithm=algorithm,options=geometryOptions,payloads=ctx.payloads,blocks={},obstacles={},fixed={},geometry={}}
    for name,h in pairs(C.blocks) do if not (ctx.payloads[side] and ctx.payloads[side].handle==h) then world.blocks[name]=pose(h) end end
    for _,h in ipairs(C.obstacles) do world.obstacles[h]=pose(h) end
    world.table=pose(C.table)
    world.geometry[C.table]=meshFingerprint(C.table)
    for _,h in ipairs(C.obstacles) do world.geometry[h]=meshFingerprint(h) end
    for _,h in pairs(C.blocks) do world.geometry[h]=meshFingerprint(h) end
    for name,arm in pairs(A) do
        world.fixed[name..'_base']=pose(arm.model)
        if name~=side then world.fixed[name]=getq(arm) end
        for _,h in ipairs(arm.shapes) do world.geometry[h]=meshFingerprint(h) end
    end
    return serialize(world)
end
local function destroyEntry(t)
    pcall(simOMPL.destroyTask,t.task)
    for _,s in ipairs(t.spaces) do pcall(simOMPL.destroyStateSpace,s) end
end
function mpInvalidate()
    for _,t in pairs(tasks) do destroyEntry(t) end tasks={}
    return {ok=true}
end
function mpCleanup()
    mpInvalidate()
    for _,arm in pairs(A) do pcall(simIK.eraseEnvironment,arm.env) if arm.ownsTarget then pcall(sim.removeObjects,{arm.target}) end end
    for _,c in ipairs(collections) do pcall(sim.destroyCollection,c) end
    A={} collections={} collectionSet={}
    return {ok=true}
end
function sysCall_cleanup() mpCleanup() end

function mpSetup(cfg)
    return transaction(function()
        C=copy(cfg)
        C.root=sim.getObject('/APEX_MR_Plus')
        C.aliases={}
        for _,h in ipairs(sim.getObjectsInTree(C.root,sim.handle_all,0)) do
            local name=sim.getObjectAlias(h)
            if not C.aliases[name] then C.aliases[name]=h end
        end
        C.table=assert(cfg.table)
        C.blocks={} C.blockHandles={}
        for name in pairs(cfg.blocks) do
            C.blocks[name]=resolve(cfg.blocks[name])
            C.blockHandles[C.blocks[name]]=name
        end
        C.obstacles={}
        for _,h in ipairs(cfg.obstacles or {}) do C.obstacles[#C.obstacles+1]=resolve(h) end
        -- Only explicitly horizontal bearing surfaces receive z-contact
        -- tolerance exemptions. A future vertical obstacle is never included
        -- merely because it is present in the obstacle list.
        C.surfaceHandles={[C.table]=true}
        for _,h in ipairs(C.obstacles) do
            local alias=sim.getObjectAlias(h)
            if alias=='CastleBaseplate' or alias=='A_SupplyTray' or alias=='B_SupplyTray' then
                C.surfaceHandles[h]=true
            end
        end
        for _,side in ipairs({'A','B'}) do
            local old=cfg.arms[side]
            local arm=copy(old)
            arm.model=assert(old.model)
            arm.tip=assert(old.tip)
            arm.target=old.target or sim.createDummy(.001)
            arm.ownsTarget=not old.target
            if arm.ownsTarget then sim.setObjectParent(arm.target,C.root,true) sim.setObjectInt32Param(arm.target,sim.objintparam_visibility_layer,0) end
            arm.joints=assert(old.joints)
            assert(#arm.joints==6,'Expected six UR5 joints')
            arm.shapes={}
            for _,h in ipairs(sim.getObjectsInTree(arm.model,sim.object_shape_type,0)) do
                local alias=sim.getObjectAlias(h)
                local explicitTool=alias:match('^'..side..'_') and (alias:find('Palm') or alias:find('Finger') or alias:find('SupportShoe') or alias:find('Tool'))
                local dynamic=sim.getObjectInt32Param(h,sim.shapeintparam_static)==0
                local respondable=sim.getObjectInt32Param(h,sim.shapeintparam_respondable)~=0
                if h==arm.model or dynamic or respondable or explicitTool then
                    arm.shapes[#arm.shapes+1]=h
                end
            end
            arm.links={}
            local jointIndex={} for i,j in ipairs(arm.joints) do jointIndex[j]=i end
            local moving,withoutTool={},{}
            for _,h in ipairs(arm.shapes) do
                local p,index=h,0
                while p~=-1 and p~=C.root do
                    if jointIndex[p] then index=jointIndex[p] break end
                    p=sim.getObjectParent(p)
                end
                arm.links[index]=arm.links[index] or {}
                table.insert(arm.links[index],h)
                if h~=arm.model then moving[#moving+1]=h end
                local alias=sim.getObjectAlias(h)
                local isTool=alias:match('^'..side..'_') and (alias:find('Palm') or alias:find('Finger') or alias:find('SupportShoe') or alias:find('Tool'))
                if not isTool then withoutTool[#withoutTool+1]=h end
            end
            arm.linkCollections={}
            for i,sh in pairs(arm.links) do arm.linkCollections[i]=collection(sh) end
            arm.selfpairs={}
            for i,one in pairs(arm.linkCollections) do for j,two in pairs(arm.linkCollections) do
                if j>i+1 then arm.selfpairs[#arm.selfpairs+1]={one,two} end
            end end
            arm.all=collection(arm.shapes)
            arm.moving=collection(moving)
            arm.withoutTool=collection(withoutTool)
            arm.env=simIK.createEnvironment()
            arm.group=simIK.createGroup(arm.env)
            simIK.setGroupCalculation(arm.env,arm.group,simIK.method_damped_least_squares,.03,80)
            local element,mapping=simIK.addElementFromScene(arm.env,arm.group,arm.model,arm.tip,arm.target,simIK.constraint_pose)
            simIK.setElementPrecision(arm.env,arm.group,element,{1e-5,1e-4})
            arm.ikjoints={} for _,j in ipairs(arm.joints) do arm.ikjoints[#arm.ikjoints+1]=mapping[j] end
            A[side]=arm
        end
        return {ok=true,algorithms={'RRTConnect','PRM','LazyPRM','PRMstar','LazyPRMstar','RRTstar','BITstar'},
            initialhome_q={A=getq(A.A),B=getq(A.B)},geometry=COLLISION_POLICY,
            collision_geometry_policy=COLLISION_POLICY,
            collision_shape_counts={A=#A.A.shapes,B=#A.B.shapes},
            state_resolution_fraction=.001}
    end)
end

function mpFK(side,q)
    return transaction(function() setq(A[side],q) return {ok=true,pose=pose(A[side].tip)} end)
end
function mpValidateJoint(side,q,payload,placed,options)
    return transaction(function()
        lastProblem=nil
        local ctx=newContext({[side]=payload},placed,options,side)
        local ok=validJoint(side,q,ctx)
        return {ok=ok,problem=lastProblem}
    end)
end
function mpValidatePair(left,right,payloads,placed,options)
    return transaction(function()
        lastProblem=nil
        local ctx=newContext(payloads,placed,options)
        setq(A.A,left) setq(A.B,right)
        for side,p in pairs(ctx.payloads) do movePayload(side,p) end
        return {ok=geometry(ctx),problem=lastProblem}
    end)
end
function mpValidateSamples(samples,options)
    return transaction(function()
        local base=snapshot() local results={}
        for _,s in ipairs(samples) do
            restore(base) lastProblem=nil
            local ctx=newContext(s.payloads or {},s.placed or {},s.options or options)
            if s.A then setq(A.A,s.A) end
            if s.B then setq(A.B,s.B) end
            for side,p in pairs(ctx.payloads) do movePayload(side,p) end
            results[#results+1]={ok=geometry(ctx),problem=lastProblem}
        end
        return results
    end)
end
function mpSolveIK(side,p,quat,payload,placed,options)
    return transaction(function()
        options=options or {} checks=0 ikCalls=0 lastProblem=nil
        local start=clock()
        local arm=A[side]
        local ctx=newContext({[side]=payload},placed,options,side)
        local seed=options.seed or getq(arm)
        local candidates={}
        local localQ=ikFromSeed(side,p,quat,seed)
        if localQ then candidates[#candidates+1]=localQ end
        if options.multiple~=false then
            sim.setObjectPose(arm.target,{p[1],p[2],p[3],table.unpack(quat)},sim.handle_world)
            simIK.syncFromSim(arm.env,{arm.group})
            local sampled=simIK.findConfigs(arm.env,arm.group,arm.ikjoints,
                {maxTime=options.ik_time or .3,findMultiple=true,findAlt=false,maxDist=options.max_dist or .7})
            for _,q in ipairs(sampled) do candidates[#candidates+1]=q end
        end
        table.sort(candidates,function(a,b)return dist(a,seed)<dist(b,seed)end)
        local accepted={} local rejections={}
        for _,q in ipairs(candidates) do
            if validJoint(side,q,ctx) then
                local duplicate=false
                for _,old in ipairs(accepted) do if dist(q,old)<.001 then duplicate=true end end
                if not duplicate then accepted[#accepted+1]=q end
                if #accepted >= (options.max_solutions or 12) then break end
            else rejections[#rejections+1]=lastProblem end
        end
        return {ok=#accepted>0,solutions=accepted,problem=#accepted==0 and lastProblem or nil,
            stats={wall_time_s=clock()-start,candidates=#candidates,collision_checks=checks,rejections=rejections,ik_calls=ikCalls}}
    end)
end

local function unpackPath(flat,dim)
    local path={} for i=1,#flat,dim do local q={} for j=0,dim-1 do q[#q+1]=flat[i+j] end path[#path+1]=q end return path
end
local function pathCost(path)
    local cost=0 for i=2,#path do cost=cost+dist(path[i],path[i-1]) end return cost
end
local function plannerData(task,dim)
    local ok,states,_,_,edges=pcall(simOMPL.getPlannerData,task)
    if ok then return {vertices=#states/dim,edges=#edges/2} end return {}
end
local function denseJoint(path,step)
    assert(type(step)=='number' and step>0 and step<math.huge,'Interpolation step must be finite and positive')
    local out={path[1]}
    for i=2,#path do
        local a,b=path[i-1],path[i]
        local n=math.max(1,math.ceil(dist(a,b)/step))
        for k=1,n do local q={} for j=1,#a do q[j]=a[j]+(b[j]-a[j])*k/n end out[#out+1]=q end
    end return out
end
local function checkedShortcut(path,check,step)
    -- A bounded greedy postprocess of an ALREADY SOLVED RRT path. Every
    -- proposed shortcut is sampled at the requested collision resolution.
    local out={path[1]}
    local at=1
    local trials=0
    while at<#path do
        local accepted=at+1
        for goal=#path,at+2,-1 do
            if trials>=24 then break end
            trials=trials+1
            local segment=denseJoint({path[at],path[goal]},step)
            local clear=true
            for _,q in ipairs(segment) do
                if not check(q) then clear=false break end
            end
            if clear then accepted=goal break end
        end
        out[#out+1]=path[accepted]
        at=accepted
    end
    return out
end
function mpPlanJoint(side,startQ,goalQ,algorithm,payload,placed,options)
    return transaction(function()
        options=options or {} checks=0 ikCalls=0 lastProblem=nil
        local startTime=clock()
        local ctx=newContext({[side]=payload},placed,options,side)
        if not validJoint(side,startQ,ctx) then return {ok=false,error='Invalid start',problem=lastProblem} end
        if not validJoint(side,goalQ,ctx) then return {ok=false,error='Invalid goal',problem=lastProblem} end
        local actualBounds={}
        for i,j in ipairs(A[side].joints) do
            local cyclic,interval=sim.getJointInterval(j)
            local low,high=interval[1],interval[1]+interval[2]
            if cyclic then low=-2*math.pi high=2*math.pi end
            local margin=options.joint_bound_margins and options.joint_bound_margins[i] or options.joint_bound_margin
            if margin then
                assert(margin>0,'joint bound margins must be positive')
                low=math.max(low,math.min(startQ[i],goalQ[i])-margin)
                high=math.min(high,math.max(startQ[i],goalQ[i])+margin)
            end
            actualBounds[i]={low,high}
        end
        local key=side..':'..algorithm..':'..serialize(goalQ)..':'..serialize(actualBounds)
        -- RRTConnect is a fresh single query here. Computing a full triangle
        -- fingerprint of every robot mesh cannot enable useful roadmap reuse
        -- and used to dominate short pick/place queries.
        local sign=algorithm=='RRTConnect' and '' or signature(side,ctx,algorithm)
        local old=tasks[key]
        if algorithm=='RRTConnect' and old then
            destroyEntry(old) tasks[key]=nil old=nil
        end
        -- Installed plugin preserves PRM/PRMstar graphs when start changes,
        -- but a post-setup setGoalState does not replace its live OMPL goal.
        -- Cache by goal; LazyPRM permits only continuation of the same query.
        local multi=algorithm=='PRM' or algorithm=='PRMstar'
        local sameQuery=old and dist(old.startQ,startQ)<1e-8
        local reused=old and old.signature==sign and (multi or sameQuery) or false
        if old and not reused then destroyEntry(old) tasks[key]=nil end
        local entry=tasks[key]
        if not entry then
            local task=simOMPL.createTask('dual_arm_'..key)
            local spaces={}
            for i,j in ipairs(A[side].joints) do
                -- Explicit [-2pi,2pi] coordinates avoid ambiguous wrap interpolation
                -- in generated executable paths; these are joint-position spaces.
                local low,high=actualBounds[i][1],actualBounds[i][2]
                spaces[#spaces+1]=simOMPL.createStateSpace(side..i,simOMPL.StateSpaceType.joint_position,j,{low},{high},i<=3 and 1 or 0)
            end
            simOMPL.setStateSpace(task,spaces)
            local algorithmId=assert(simOMPL.Algorithm[algorithm],'Planner unavailable: '..algorithm)
            simOMPL.setAlgorithm(task,algorithmId)
            simOMPL.setStateValidityCheckingResolution(task,options.resolution or .001)
            simOMPL.setVerboseLevel(task,0)
            simOMPL.setStateValidationCallback(task,function(q)return validJoint(side,q,context)end)
            entry={task=task,spaces=spaces,signature=sign,setup=false,startQ=copy(startQ)}
            tasks[key]=entry
        end
        context=ctx
        simOMPL.setStartState(entry.task,startQ)
        simOMPL.setGoalState(entry.task,goalQ)
        if not entry.setup then simOMPL.setup(entry.task) entry.setup=true end
        entry.startQ=copy(startQ)
        local before=plannerData(entry.task,6)
        local checksBeforeSolve=checks
        local solved=simOMPL.solve(entry.task,options.solve_time or 1.5)
        local exact=solved and simOMPL.hasExactSolution(entry.task)
        local stats={algorithm=algorithm,backend='simOMPL',roadmap_reused=reused and multi,
            planner_reused=reused,planner_before=before,joint_bounds=actualBounds,
            joint_bound_margin=options.joint_bound_margin,joint_bound_margins=options.joint_bound_margins,
            planner_after=plannerData(entry.task,6),solve_budget_s=options.solve_time or 1.5}
        stats.solve_collision_checks=checks-checksBeforeSolve
        if not exact then
            stats.wall_time_s=clock()-startTime stats.collision_checks=checks
            return {ok=false,error='No exact collision-free solution within budget',stats=stats,problem=lastProblem}
        end
        local raw=unpackPath(simOMPL.getPath(entry.task),6)
        local simplified=raw
        if options.ompl_simplify_time and options.ompl_simplify_time>0 then
            simOMPL.simplifyPath(entry.task,options.ompl_simplify_time)
            local candidate=unpackPath(simOMPL.getPath(entry.task),6)
            if pathCost(candidate)<pathCost(raw) then simplified=candidate end
        end
        local extent=0
        for _,bound in ipairs(actualBounds) do extent=extent+(bound[2]-bound[1])^2 end
        local checkStep=math.min(.005,(options.resolution or .001)*math.sqrt(extent))
        local waypoints=simplified
        local checksBeforeShortcut=checks
        if options.simplify~=false then
            waypoints=checkedShortcut(simplified,function(q)return validJoint(side,q,ctx)end,checkStep)
        end
        stats.postprocess=options.ompl_simplify_time and 'OMPL_simplification_and_sampled_shortcuts' or 'bounded_collision_checked_joint_shortcut'
        stats.ompl_simplify_time_s=options.ompl_simplify_time or 0
        stats.simplified_cost=pathCost(simplified)
        stats.shortcut_collision_checks=checks-checksBeforeShortcut
        stats.collision_check_step_rad=checkStep
        local certificate=denseJoint(waypoints,checkStep)
        for i,q in ipairs(certificate) do
            if not validJoint(side,q,ctx) then return {ok=false,error='Dense path validation failed',sample=i,problem=lastProblem} end
        end
        local path=denseJoint(waypoints,options.max_joint_step or .04)
        if dist(path[1],startQ)>1e-5 or dist(path[#path],goalQ)>1e-5 then return {ok=false,error='Planner endpoint mismatch'} end
        stats.wall_time_s=clock()-startTime stats.collision_checks=checks
        stats.raw_cost=pathCost(raw) stats.raw_states=#raw stats.validated_states=#certificate
        stats.execution_states=#path
        return {ok=true,path=path,cost=pathCost(path),stats=stats}
    end)
end


-- Validate shortcut edges using the exact same payload/contact context as RRT.
function mpValidateJointEdge(side,startQ,goalQ,payload,placed,options)
    return transaction(function()
        options=options or {} checks=0 lastProblem=nil
        local ctx=newContext({[side]=payload},placed,options,side)
        local samples=denseJoint({startQ,goalQ},options.edge_check_step or .005)
        for i,q in ipairs(samples) do
            if not validJoint(side,q,ctx) then
                return {ok=false,sample=i,samples=#samples,problem=lastProblem,collision_checks=checks}
            end
        end
        return {ok=true,samples=#samples,collision_checks=checks}
    end)
end
