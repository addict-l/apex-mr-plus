"""Native full-state transactions for the frozen LEGO workcell predicate."""
BATCH_GEOMETRY = r"""
local armsConfig,partsConfig={},{}
local padMeshes={}
local function matrixError(a,b)
    local sum,trace=0,0
    for i=1,12 do
        if i%4==0 then sum=sum+(a[i]-b[i])^2 else trace=trace+a[i]*b[i] end
    end
    return {math.sqrt(sum),math.acos(math.max(-1,math.min(1,(trace-1)*.5)))}
end
local function matrixCheck(a,b,location,positionTolerance)
    local err=matrixError(a,b)
    if err[1]>(positionTolerance or 1e-5) or err[2]>2e-4 then
        return {ok=false,kind=location,position_error_m=err[1],angular_error_rad=err[2]}
    end
    return nil
end
local function sampleWorld()
    local row={q={},jaw={},follower={},parts={}}
    for side,arm in pairs(armsConfig) do
        row.q[side]=bridgeReadJoints(arm.joints)
        row.jaw[side]=sim.getJointPosition(arm.master)
        row.follower[side]=sim.getJointPosition(arm.follower)
    end
    for name,part in pairs(partsConfig) do
        local h=part.handle
        local parent=sim.getObjectParent(h)
        local data={pose=sim.getObjectMatrix(h,sim.handle_world),parent=parent}
        for side,arm in pairs(armsConfig) do
            if parent==arm.tcp then
                data.owner=side
                data.binding=sim.getObjectMatrix(h,parent)
            end
        end
        row.parts[name]=data
    end
    return row
end
function bridgeBatchConfigure(arms,parts)
    armsConfig=arms
    partsConfig=parts
    for _,arm in pairs(armsConfig) do
        for _,handle in ipairs(arm.pads) do
            local vertices,triangles=sim.getShapeMesh(handle)
            assert(#vertices>0 and #triangles>0,'Contact geometry mesh was empty')
            padMeshes[handle]={vertices,triangles}
        end
    end
    return sampleWorld()
end
function bridgeSnapshot()
    return sampleWorld()
end
local function applyJoints(state)
    for side,arm in pairs(armsConfig) do
        local q=bridgeWriteJoints(arm.joints,state.q[side])
        for i,v in ipairs(q) do
            assert(math.abs(v-state.q[side][i])<=1e-7,'Joint readback changed')
        end
        local j=bridgeWriteJaw(arm.master,arm.follower,state.jaw[side],arm.direct)
        assert(math.abs(j[1]-state.jaw[side])<=1e-7,'Jaw readback changed')
        local expected=(state.follower or {})[side] or state.jaw[side]
        assert(math.abs(j[2]-expected)<=1e-7,'Jaw follower readback changed')
    end
end
local function restoreWorld(state)
    applyJoints(state)
    for name,part in pairs(partsConfig) do
        local target=state.parts[name]
        local h=part.handle
        if sim.getObjectParent(h)~=target.parent then sim.setObjectParent(h,target.parent,true) end
        sim.setObjectMatrix(h,target.pose,sim.handle_world)
        local actual=sim.getObjectMatrix(h,sim.handle_world)
        for i=1,12 do assert(math.abs(actual[i]-target.pose[i])<=1e-8,'World matrix restoration failed') end
    end
end
local function applyWorld(state,enforce)
    applyJoints(state)
    local shifts={}
    for name,part in pairs(partsConfig) do
        local h=part.handle
        local target=state.parts[name]
        local owner=target.owner
        local parent=owner and armsConfig[owner].tcp or part.parent
        local old=sim.getObjectParent(h)
        if enforce then
            -- With q already applied while still parented to the old holder,
            -- the source pose is the physical pre-transfer pose.
            local before=sim.getObjectMatrix(h,sim.handle_world)
            local destination=owner and sim.multiplyMatrices(sim.getObjectMatrix(parent,sim.handle_world),target.binding)
                                  or target.pose
            local mismatch=matrixCheck(before,destination,'binding_mismatch')
            if mismatch then mismatch.part=name; return mismatch end
            if old~=parent then
                sim.setObjectParent(h,parent,true)
                local after=sim.getObjectMatrix(h,sim.handle_world)
                local error=matrixError(before,after)
                if error[1]>1e-5 or error[2]>2e-4 then
                    return {ok=false,kind='transfer_world_pose_jump',part=name,
                            position_error_m=error[1],angular_error_rad=error[2]}
                end
                shifts[#shifts+1]={part=name,old_parent=old,new_parent=parent,
                                    position_error_m=error[1],angular_error_rad=error[2]}
            end
        else
            if old~=parent then sim.setObjectParent(h,parent,true) end
            if owner then
                sim.setObjectMatrix(h,target.binding,parent)
            else
                sim.setObjectMatrix(h,target.pose,sim.handle_world)
            end
        end
        if sim.getObjectParent(h)~=parent then
            return {ok=false,kind='attachment_parent_mismatch',part=name}
        end
    end
    local fk={}
    for side,arm in pairs(armsConfig) do
        local m=sim.getObjectMatrix(arm.tcp,sim.handle_world)
        fk[side]=m
        local expected=(state.tcp_targets or {})[side]
        if expected then
            local mismatch=matrixCheck(m,expected,'native_fk_mismatch',5e-5)
            if mismatch then mismatch.arm=side; return mismatch end
        end
    end
    return {ok=true,tcp=fk,transfers=shifts}
end
function bridgeApplyState(state,enforce)
    local previous=sampleWorld()
    local ok,result=xpcall(function() return applyWorld(state,enforce) end,debug.traceback)
    if not ok or not result.ok then
        restoreWorld(previous)
        if not ok then return {ok=false,kind='native_state_error',error=result,restored=true} end
        result.restored=true
    end
    return result
end
local function point(inMatrix,v)
    return {inMatrix[1]*v[1]+inMatrix[2]*v[2]+inMatrix[3]*v[3]+inMatrix[4],
            inMatrix[5]*v[1]+inMatrix[6]*v[2]+inMatrix[7]*v[3]+inMatrix[8],
            inMatrix[9]*v[1]+inMatrix[10]*v[2]+inMatrix[11]*v[3]+inMatrix[12]}
end
local function clip(polygon,axis,bound,greater)
    if #polygon==0 then return polygon end
    local result={}
    local prev=polygon[#polygon]
    local inside=(greater and prev[axis]>=bound) or (not greater and prev[axis]<=bound)
    for _,curr in ipairs(polygon) do
        local now=(greater and curr[axis]>=bound) or (not greater and curr[axis]<=bound)
        if now~=inside then
            local t=(bound-prev[axis])/(curr[axis]-prev[axis])
            result[#result+1]={prev[1]+t*(curr[1]-prev[1]),prev[2]+t*(curr[2]-prev[2]),
                               prev[3]+t*(curr[3]-prev[3])}
        end
        if now then result[#result+1]=curr end
        prev,inside=curr,now
    end
    return result
end
local function wallSide(handle,frame,dim,axis)
    local across=3-axis
    local pad=padMeshes[handle]
    local tr=sim.multiplyMatrices(frame,sim.getObjectMatrix(handle,sim.handle_world))
    local localPoints,sum,minz={},0,math.huge
    for i=1,#pad[1],3 do
        local v=point(tr,{pad[1][i],pad[1][i+1],pad[1][i+2]})
        localPoints[#localPoints+1]=v
        sum=sum+v[axis]
        minz=math.min(minz,v[3])
    end
    local sign=sum>0 and 1 or -1
    local clearance=math.huge
    local w=dim[axis]/2
    local halfAcross=dim[across]/2
    for _,v in ipairs(localPoints) do clearance=math.min(clearance,sign*v[axis]-w) end
    local area=0
    local zlo,zhi=math.huge,-math.huge
    if clearance>=-1e-8 and clearance<=.0001 then
        for i=1,#pad[2],3 do
            local polygon={localPoints[pad[2][i]+1],localPoints[pad[2][i+1]+1],localPoints[pad[2][i+2]+1]}
            polygon=clip(polygon,axis,sign*(w-1e-8),sign>0)
            polygon=clip(polygon,axis,sign*(w+.0001),sign<0)
            polygon=clip(polygon,across,-halfAcross,true)
            polygon=clip(polygon,across,halfAcross,false)
            polygon=clip(polygon,3,0,true)
            polygon=clip(polygon,3,dim[3],false)
            if #polygon>=3 then
                local signed=0
                for j,v in ipairs(polygon) do
                    local nextPoint=polygon[j%#polygon+1]
                    signed=signed+v[across]*nextPoint[3]-v[3]*nextPoint[across]
                end
                local a=math.abs(signed)*.5
                if a>1e-13 then
                    area=area+a
                    for _,v in ipairs(polygon) do
                        zlo=math.min(zlo,v[3]); zhi=math.max(zhi,v[3])
                    end
                end
            end
        end
    end
    local height=area>0 and zhi-zlo or 0
    return {sign=sign,valid=clearance>=-1e-8 and clearance<=.0001 and area>1e-10,
            pad=handle,wall_clearance_m=clearance,overlap_area_m2=area,
            pad_tip_height_from_bottom_m=minz,contact_height_m=height}
end
local function checkFaces(state)
    local result,allowed={},{}
    for _,pair in ipairs(state.allow or {}) do allowed[#allowed+1]=pair end
    for _,request in ipairs(state.faces or {}) do
        local part=assert(partsConfig[request.part],'Unknown contact part')
        local arm=assert(armsConfig[request.side],'Unknown contact arm')
        local axis=request.axis+1
        assert(axis==1 or axis==2,'Invalid contact axis')
        local bottom=sim.multiplyMatrices(sim.getObjectMatrix(part.handle,sim.handle_world),part.mesh_to_bottom)
        local frame=sim.getMatrixInverse(bottom)
        local left=wallSide(arm.pads[1],frame,part.dimensions,axis)
        local right=wallSide(arm.pads[2],frame,part.dimensions,axis)
        result[#result+1]={part=request.part,side=request.side,axis=request.axis,pads={left,right}}
        if not left.valid or not right.valid or left.sign==right.sign then
            return result,nil
        end
        allowed[#allowed+1]={left.pad,part.handle}
        allowed[#allowed+1]={right.pad,part.handle}
    end
    return result,allowed
end
function bridgeBatchSamples(states,limit,stopFirst,workSeconds)
    -- A call is a fully restoring transaction even if an input, collision
    -- callback, or cooperative time guard fails.
    local initial=sampleWorld()
    local started=sim.getSystemTime()
    local rows={}
    local result={rows=rows,ok=true,timing={state_fk_seconds=0,face_seconds=0,
                   geometry_seconds=0,collision_seconds=0,distance_seconds=0,aabb_seconds=0}}
    local done,problem=xpcall(function()
        for i,state in ipairs(states) do
            if i>1 and sim.getSystemTime()-started>=workSeconds then
                result.paused_for_batch_budget=true
                break
            end
            local start=sim.getSystemTime()
            local applied=applyWorld(state,false)
            result.timing.state_fk_seconds=result.timing.state_fk_seconds+(sim.getSystemTime()-start)
            local row=applied
            if applied.ok then
                start=sim.getSystemTime()
                local faces,allowed=checkFaces(state)
                result.timing.face_seconds=result.timing.face_seconds+(sim.getSystemTime()-start)
                if not allowed then
                    row={ok=false,kind='pad_face_mismatch',faces=faces}
                else
                    start=sim.getSystemTime()
                    row=bridgeSample(allowed,limit,state.measure_distance~=false,true,
                                     state.filtered_clearance==true,state.all_collisions==true)
                    result.timing.geometry_seconds=result.timing.geometry_seconds+(sim.getSystemTime()-start)
                    result.timing.collision_seconds=result.timing.collision_seconds+row.timing.collision_seconds
                    result.timing.distance_seconds=result.timing.distance_seconds+row.timing.distance_seconds
                    result.timing.aabb_seconds=result.timing.aabb_seconds+row.timing.aabb_seconds
                    row.faces=faces
                end
                row.tcp=applied.tcp
            end
            rows[#rows+1]=row
            if not row.ok then
                result.ok=false
                if not result.first_failure then result.first_failure=i end
                if stopFirst then break end
            end
        end
    end,debug.traceback)
    local before=sim.getSystemTime()
    restoreWorld(initial)
    result.timing.restore_seconds=sim.getSystemTime()-before
    result.timing.native_seconds=sim.getSystemTime()-started
    result.restored=true
    if not done then result.ok=false; result.error=problem end
    return result
end
"""
