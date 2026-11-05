sim = require 'sim'
local simUI
if not sim.getBoolParam(sim.boolparam_headless) then simUI = require 'simUI' end
local ui, cfg, root, objects

local function visibility(alias, visible)
    for _, h in ipairs(sim.getObjectsInTree(objects[alias], sim.handle_all, 0)) do
        if sim.getObjectType(h) == sim.object_shape_type then
            sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, visible and 1 or 0)
        end
    end
end

function initialize()
    root = sim.getObject('/APEX_MR_Scene3')
    cfg = sim.readCustomTableData(root, 'Scene3')
    objects = {}
    for _, h in ipairs(sim.getObjectsInTree(root, sim.handle_all, 0)) do
        objects[sim.getObjectAlias(h)] = h
    end
    if simUI and not ui then
        ui = simUI.create([[<ui title="APEX MR Scene 3 | 临时支撑与机械臂释放" closeable="false" resizable="false" placement="absolute" position="40,100" layout="vbox">
          <label text="双 UR5 · 可拆支撑的悬挑桥面装配" style="font-size:16px;font-weight:bold;"/>
          <label text="橙色 A：扶持 / 左侧供料　蓝色 B：支撑 / 安装"/>
          <button text="① A 托住悬梁：供料资源被占用" on-click="initial"/>
          <button text="② 外侧支撑承载：A 取专属零件" on-click="released"/>
          <button text="③ 永久连接闭合：准备卸载支撑" on-click="closed"/>
          <button text="④ 支撑回仓：封板后结构完成" on-click="complete"/>
          <button text="⑤ 对照：内侧支撑被封板困住" on-click="trapped"/>
          <group layout="hbox" flat="true"><button text="全景" on-click="overview"/><button text="近景" on-click="detail"/></group>
          <label id="10" text="" wordwrap="true" style="max-width:440px;color:#21485b;"/>
          <label text="当前为设计状态预览：按钮直接切换物体与关节位姿。"/>
          <label text="承载条件是待验证要求；尚未执行 RRT / 力控 / 动力学。"/>
        </ui>]])
    end
end

function setPreset(mode)
    if not cfg then initialize() end
    if sim.getSimulationState() ~= sim.simulation_stopped then return false end
    if mode < 0 or mode > 4 then return false end
    local state = cfg.states[mode+1]
    for _, side in ipairs({'A','B'}) do
        local q = cfg.poses[state[side]]
        for i=1,6 do sim.setJointPosition(objects[side..'_joint_'..i], q[i]) end
    end
    for alias, pose in pairs(state.objects) do
        sim.setObjectPose(objects[alias], pose, sim.handle_world)
    end
    visibility('A_SupportPad', mode == 0)
    visibility('OuterGuide', mode ~= 4)
    visibility('InnerGuide', mode == 4)
    if ui then simUI.setLabelText(ui, 10, state.description) end
    cfg.preset=mode
    sim.writeCustomTableData(root,'Scene3',cfg)
    return true
end
function initial() setPreset(0) end
function released() setPreset(1) end
function closed() setPreset(2) end
function complete() setPreset(3) end
function trapped() setPreset(4) end
function overview() sim.setObjectPose(sim.getObject('/DefaultCamera'), cfg.overview_pose, sim.handle_world) end
function detail() sim.setObjectPose(sim.getObject('/DefaultCamera'), cfg.detail_pose, sim.handle_world) end
function getPreviewState() return {preset=cfg.preset, kind='static_design_preview', dynamics_validated=false, motion_planner_connected=false, ui_created=ui ~= nil} end
function sysCall_init() initialize() end
function sysCall_beforeSimulation()
    if ui then simUI.setLabelText(ui,10,'这是静态设计预览。请停止仿真，使用按钮切换状态。') end
end
function sysCall_cleanup() if ui then simUI.destroy(ui) ui=nil end end
