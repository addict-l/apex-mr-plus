-- Type-2 TPG oracle only: no environment, self, or free-part collision checks.
-- All mutations are restored once per batch, including exceptional exits.
sim=require 'sim'
local C,A,aliases,blocks,collections,collectionObjects={},{},{},{},{},{}
local blockShapes,blockCollision,blockMembers,articulations={},{},{},{}
local checks,batches=0,0
local boundsCalls,localCorners=0,{}
local POLICY='native base + dynamic/respondable collision shapes + explicit cuboid Palm/Finger/SupportShoe tools; purely visual static shells excluded'
local SCOPE='interarm robot proxies, carried payload vs opposite robot proxies, distinct carried payload pairs only'

function sysCall_init() end

local function copy(t)
    local out={} for k,v in pairs(t or {}) do out[k]=v end return out
end
local function pose(h) return sim.getObjectPose(h,sim.handle_world) end
local function getq(arm)
    local q={} for i,j in ipairs(arm.joints) do q[i]=sim.getJointPosition(j) end return q
end
local function setq(arm,q)
    assert(type(q)=='table' and #q==6,'Expected six joint positions')
    for i,j in ipairs(arm.joints) do sim.setJointPosition(j,q[i]) end
end
local function resolve(value)
    if type(value)=='number' then return value end
    assert(type(value)=='string','Object reference must be an alias/path or handle')
    if blocks[value] then return blocks[value] end
    if aliases[value] then return aliases[value] end
    if value:sub(1,1)=='/' then return sim.getObject(value) end
    return sim.getObject((C.root_path or '/APEX_MR_Scene3')..'/'..value)
end
local function collection(shapes)
    local c=sim.createCollection(1)
    for _,h in ipairs(shapes) do sim.addItemToCollection(c,sim.handle_single,h,0) end
    collections[#collections+1]=c
    collectionObjects[c]=shapes
    return c
end
local function explicitTool(side,h)
    local alias=sim.getObjectAlias(h)
    return alias:match('^'..side..'_') and
      (alias:find('Palm') or alias:find('Finger') or alias:find('SupportShoe') or alias:find('Tool'))
end
local function corners(h)
    if localCorners[h] then return localCorners[h] end
    -- A relocated compound jack's shape origin is at its foot, whereas the
    -- legacy object bbox coordinates are centered on a separate bbox frame.
    -- Derive reference-frame bounds from the actual collision mesh instead.
    local vertices=sim.getShapeMesh(h)
    local low,high={math.huge,math.huge,math.huge},{-math.huge,-math.huge,-math.huge}
    for i=1,#vertices,3 do
        for axis=1,3 do
            low[axis]=math.min(low[axis],vertices[i+axis-1])
            high[axis]=math.max(high[axis],vertices[i+axis-1])
        end
    end
    assert(low[1]<math.huge,'Collision shape has no mesh vertices')
    local points={}
    for _,x in ipairs({low[1],high[1]}) do
        for _,y in ipairs({low[2],high[2]}) do
            for _,z in ipairs({low[3],high[3]}) do points[#points+1]={x,y,z} end
        end
    end
    localCorners[h]=points
    return points
end

function fastSetup(cfg)
    C=cfg; aliases={};blocks={};A={}
    local root=cfg.root or sim.getObject(cfg.root_path or '/APEX_MR_Scene3')
    for _,h in ipairs(sim.getObjectsInTree(root,sim.handle_all,0)) do
        local alias=sim.getObjectAlias(h)
        if not aliases[alias] then aliases[alias]=h end
    end
    local blockSet={}
    blockShapes={};blockCollision={};blockMembers={};articulations={}
    for name,h in pairs(cfg.blocks or {}) do
        blocks[name]=resolve(h)
        local shapes={}
        for _,raw in ipairs((cfg.block_shapes or {})[name] or sim.getObjectsInTree(blocks[name],sim.object_shape_type,0)) do
            local sh=resolve(raw)
            if (cfg.block_shapes or {})[name] or (sim.getObjectSpecialProperty(sh)&sim.objectspecialproperty_collidable)~=0 then
                shapes[#shapes+1]=sh;blockSet[sh]=true
            end
        end
        assert(#shapes>0,'Object has no collision shapes: '..name)
        blockShapes[blocks[name]]=shapes;blockCollision[blocks[name]]=collection(shapes)
        blockMembers[blocks[name]]={}
        for _,sh in ipairs(shapes) do blockMembers[blocks[name]][sh]=true end
    end
    for name,h in pairs(cfg.articulations or {}) do articulations[name]=resolve(h) end
    local shapeCounts={}
    for _,side in ipairs({'A','B'}) do
        local model=resolve(cfg.arms[side].model)
        local arm={model=model,tip=resolve(cfg.arms[side].tip),joints={},shapes={}}
        for i,j in ipairs(cfg.arms[side].joints) do arm.joints[i]=resolve(j) end
        assert(#arm.joints==6,'Expected six UR5 joints')
        for _,h in ipairs(sim.getObjectsInTree(model,sim.object_shape_type,0)) do
            local dynamic=sim.getObjectInt32Param(h,sim.shapeintparam_static)==0
            local respondable=sim.getObjectInt32Param(h,sim.shapeintparam_respondable)~=0
            if not blockSet[h] and (h==model or dynamic or respondable or explicitTool(side,h)) then
                arm.shapes[#arm.shapes+1]=h
            end
        end
        arm.all=collection(arm.shapes)
        for _,h in ipairs(arm.shapes) do corners(h) end
        A[side]=arm;shapeCounts[side]=#arm.shapes
    end
    return {ok=true,scope=SCOPE,collision_geometry_policy=POLICY,
            collision_shape_counts=shapeCounts,environment_checked=false,
            self_collision_checked=false,free_parts_checked=false,
            placed_snapshots_used=false,transaction='one snapshot/restore per batch',
            swept_bounds='proxy local bbox corner FK union plus revolute-joint radius inflation',
            swept_bounds_model_radius_m=2.0,swept_bounds_radius_is_assumption=true}
end

local function snapshot(samples)
    local saved={arms={},payloads={},articulations={}}
    for side,arm in pairs(A) do saved.arms[side]={q=getq(arm),tip=pose(arm.tip)} end
    for _,sample in ipairs(samples) do
        for _,raw in pairs(sample.payloads or {}) do
            local p=type(raw)=='string' and {name=raw} or raw
            local name=p and (p.name or p.part)
            if name then
                local h=resolve(name)
                if not saved.payloads[h] then saved.payloads[h]=pose(h) end
            end
        end
    end
    for name,h in pairs(articulations) do saved.articulations[name]=sim.getJointPosition(h) end
    return saved
end
local function restore(saved)
    for side,state in pairs(saved.arms) do setq(A[side],state.q) end
    for h,p in pairs(saved.payloads) do sim.setObjectPose(h,p,sim.handle_world) end
    for name,q in pairs(saved.articulations or {}) do sim.setJointPosition(articulations[name],q) end
end
local function payloadSpec(side,raw,saved)
    if type(raw)=='string' then raw={name=raw} end
    local name=raw and (raw.name or raw.part)
    if not name then return nil end
    local p=copy(raw);p.name=name;p.handle=resolve(name)
    if not p.offset then
        local relative=sim.multiplyPoses(sim.getPoseInverse(saved.arms[side].tip),saved.payloads[p.handle])
        p.offset={relative[1],relative[2],relative[3]}
        p.quaternion={relative[4],relative[5],relative[6],relative[7]}
    end
    p.quaternion=p.quaternion or {0,0,0,1}
    assert(#p.offset==3 and #p.quaternion==4,'Invalid payload transform')
    return p
end
local function payloadPose(side,p)
    return sim.multiplyPoses(pose(A[side].tip),{p.offset[1],p.offset[2],p.offset[3],table.unpack(p.quaternion)})
end
local function exclusions(options)
    local result={}
    for _,pair in ipairs((options or {}).contact_exclusions or {}) do
        assert(#pair==2,'Contact exclusions must be exact pairs')
        local a,b=resolve(pair[1]),resolve(pair[2])
        for _,one in ipairs(blockShapes[a] or {a}) do for _,two in ipairs(blockShapes[b] or {b}) do
            result[one]=result[one] or {};result[two]=result[two] or {}
            result[one][two]=true;result[two][one]=true
        end end
    end
    return result
end
local function excluded(a,b,allowed)
    return allowed[a] and allowed[a][b]
end
local function hit(a,b,allowed,reason)
    local collision,pair=sim.checkCollision(a,b)
    if collision==0 then return nil end
    if pair and excluded(pair[1],pair[2],allowed) then
        local one=collectionObjects[a] or {a}
        local two=collectionObjects[b] or {b}
        for _,h in ipairs(one) do for _,k in ipairs(two) do
            if h~=k and not excluded(h,k,allowed) and sim.checkCollision(h,k)>0 then
                return {reason=reason,pair={sim.getObjectAlias(h,1),sim.getObjectAlias(k,1)}}
            end
        end end
        return nil
    end
    local names={}
    for _,h in ipairs(pair or {}) do names[#names+1]=sim.getObjectAlias(h,1) end
    return {reason=reason,pair=names}
end
local function poseDifference(a,b)
    local d=0 for i=1,3 do d=d+(a[i]-b[i])^2 end
    local dot=0 for i=4,7 do dot=dot+a[i]*b[i] end
    return math.sqrt(d),2*math.acos(math.min(1,math.abs(dot)))
end
local function validate(sample,defaultOptions,saved,specCache,exclusionCache)
    checks=checks+1
    local options=sample.options or defaultOptions or {}
    for name,h in pairs(articulations) do
        sim.setJointPosition(h,(options.articulations or {})[name] or saved.articulations[name])
    end
    setq(A.A,sample.A or sample.qA or saved.arms.A.q)
    setq(A.B,sample.B or sample.qB or saved.arms.B.q)
    local specs={}
    for side,raw in pairs(sample.payloads or {}) do
        assert(A[side],'Invalid payload owner')
        -- Samples from a TPG fragment pair reuse the same small transform
        -- tables; cache by table identity where possible without changing the
        -- meaning of separately supplied bindings.
        specCache[side]=specCache[side] or {}
        local p=specCache[side][raw]
        if not p then p=payloadSpec(side,raw,saved);specCache[side][raw]=p end
        if p then specs[side]=p end
    end
    local moved={}
    for side,p in pairs(specs) do
        local predicted=payloadPose(side,p)
        if moved[p.handle] then
            local pe,ae=poseDifference(moved[p.handle],predicted)
            if pe>.002 or ae>.02 then
                return {ok=false,problem={reason='inconsistent_shared_payload',part=p.name,error_m=pe,error_rad=ae},scope=SCOPE}
            end
        else
            sim.setObjectPose(p.handle,predicted,sim.handle_world)
            moved[p.handle]=predicted
        end
    end
    local allowed=exclusionCache[options]
    if not allowed then allowed=exclusions(options);exclusionCache[options]=allowed end
    local problem=hit(A.A.all,A.B.all,allowed,'interarm_collision')
    if problem then return {ok=false,problem=problem,scope=SCOPE} end
    for side,p in pairs(specs) do
        local other=side=='A' and 'B' or 'A'
        problem=hit(blockCollision[p.handle] or p.handle,A[other].all,allowed,'payload_other_arm_collision:'..p.name..':'..other)
        if problem then return {ok=false,problem=problem,scope=SCOPE} end
    end
    if specs.A and specs.B and specs.A.handle~=specs.B.handle then
        problem=hit(blockCollision[specs.A.handle] or specs.A.handle,blockCollision[specs.B.handle] or specs.B.handle,allowed,'payload_payload_collision')
        if problem then return {ok=false,problem=problem,scope=SCOPE} end
    end
    return {ok=true,scope=SCOPE}
end

function fastValidateSamples(samples,options)
    assert(next(A),'Call fastSetup before geometry queries')
    local saved=snapshot(samples)
    local ok,results=xpcall(function()
        local answer,specCache,exclusionCache={},{},{}
        for _,sample in ipairs(samples) do answer[#answer+1]=validate(sample,options,saved,specCache,exclusionCache) end
        return answer
    end,debug.traceback)
    local restored,errorMessage=pcall(restore,saved)
    batches=batches+1
    if not restored then return {ok=false,error='Batch restore failed: '..tostring(errorMessage),scope=SCOPE} end
    if not ok then return {ok=false,error=tostring(results),scope=SCOPE} end
    return results
end

function fastStatistics()
    return {states=checks,batches=batches,bounds_calls=boundsCalls,scope=SCOPE,collision_geometry_policy=POLICY,
            environment_checked=false,self_collision_checked=false,free_parts_checked=false,
            placed_snapshots_used=false}
end

function fastNodeBounds(side,qPath,payload,modelBound,options)
    assert(A[side],'Invalid arm for node bounds')
    assert(type(qPath)=='table' and #qPath>0,'Node bounds require a nonempty q path')
    modelBound=modelBound or 2.0
    assert(modelBound>0,'Model radius bound must be positive')
    local saved=snapshot({{payloads={[side]=payload or {}}}})
    local ok,result=xpcall(function()
        for name,q in pairs((options or {}).articulations or {}) do
            sim.setJointPosition(assert(articulations[name],'Unknown articulation: '..name),q)
        end
        local part=payloadSpec(side,payload,saved)
        local low={math.huge,math.huge,math.huge}
        local high={-math.huge,-math.huge,-math.huge}
        local observedRadius=0
        local pointCount=0
        local maxDelta=0
        for i=2,#qPath do
            local delta=0
            for j=1,6 do delta=delta+math.abs(qPath[i][j]-qPath[i-1][j]) end
            maxDelta=math.max(maxDelta,delta)
        end
        for _,q in ipairs(qPath) do
            setq(A[side],q)
            if part then sim.setObjectPose(part.handle,payloadPose(side,part),sim.handle_world) end
            local jointOrigins={}
            for _,j in ipairs(A[side].joints) do jointOrigins[#jointOrigins+1]=sim.getObjectPosition(j,sim.handle_world) end
            local function addShape(h)
                local matrix=sim.getObjectMatrix(h,sim.handle_world)
                for _,p in ipairs(corners(h)) do
                    local world={
                        matrix[1]*p[1]+matrix[2]*p[2]+matrix[3]*p[3]+matrix[4],
                        matrix[5]*p[1]+matrix[6]*p[2]+matrix[7]*p[3]+matrix[8],
                        matrix[9]*p[1]+matrix[10]*p[2]+matrix[11]*p[3]+matrix[12]}
                    pointCount=pointCount+1
                    for axis=1,3 do low[axis]=math.min(low[axis],world[axis]);high[axis]=math.max(high[axis],world[axis]) end
                    for _,origin in ipairs(jointOrigins) do
                        local squared=0 for axis=1,3 do squared=squared+(world[axis]-origin[axis])^2 end
                        observedRadius=math.max(observedRadius,math.sqrt(squared))
                    end
                end
            end
            for _,h in ipairs(A[side].shapes) do addShape(h) end
            if part then for _,sh in ipairs(blockShapes[part.handle] or {part.handle}) do addShape(sh) end end
        end
        assert(observedRadius<=modelBound+1e-8,
               'Observed robot/tool/payload corner exceeds assumed revolute-joint point radius; disable broad phase or supply a proven larger bound')
        -- For any point on a linear joint segment, choose the nearer sampled
        -- endpoint. Each revolute-coordinate change moves that point by at
        -- most modelBound*abs(dq). The nearer endpoint is <= L1(dq)/2 away.
        -- Inflating the union of endpoint AABBs therefore covers the segment.
        local inflation=modelBound*maxDelta/2
        local numericalPad=1e-9
        local inflatedLow,inflatedHigh={},{}
        for axis=1,3 do
            inflatedLow[axis]=low[axis]-inflation-numericalPad
            inflatedHigh[axis]=high[axis]+inflation+numericalPad
        end
        return {ok=true,min=inflatedLow,max=inflatedHigh,raw_min=low,raw_max=high,
                inflation=inflation,numerical_padding_m=numericalPad,
                model_bound_m=modelBound,max_adjacent_sum_abs_dq=maxDelta,
                max_observed_corner_joint_distance_m=observedRadius,
                joint_samples=#qPath,world_bbox_corners_evaluated=pointCount,
                payload_included=part~=nil,arm=side,scope=SCOPE,
                collision_geometry_policy=POLICY,
                covers_between_linear_joint_samples=true,
                radius_bound_is_model_assumption=true,
                radius_bound_assumption='all native UR5 collision proxies, fixed tools and this payload stay within model_bound_m of each influencing revolute joint over the entire segment',
                scope_note='nominal rigid FK geometry only; no environment, self-collision, free parts, contact compliance or tracking error'}
    end,debug.traceback)
    local restored,errorMessage=pcall(restore,saved)
    boundsCalls=boundsCalls+1
    if not restored then return {ok=false,error='Bounds restore failed: '..tostring(errorMessage),scope=SCOPE} end
    if not ok then return {ok=false,error=tostring(result),scope=SCOPE} end
    return result
end
function fastCleanup()
    for _,c in ipairs(collections) do pcall(sim.destroyCollection,c) end
    collections={};collectionObjects={};localCorners={};A={}
    return true
end
function sysCall_cleanup() fastCleanup() end
