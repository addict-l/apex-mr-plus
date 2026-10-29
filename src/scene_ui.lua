sim = require 'sim'
local simUI
if not sim.getBoolParam(sim.boolparam_headless) then simUI = require 'simUI' end
local ui, root, cfg, objects

local function byAlias(alias)
    return objects[alias]
end

function initialize()
    root = sim.getObject('/APEX_MR_Plus')
    cfg = sim.readCustomTableData(root, 'APEX_MR_Plus')
    objects = {}
    for _, h in ipairs(sim.getObjectsInTree(root, sim.handle_all, 0)) do
        objects[sim.getObjectAlias(h)] = h
    end
    if simUI and not ui then
        local xml = [[<ui title="APEX MR Plus | 场景预览" closeable="false" resizable="false" placement="absolute" position="60,140" layout="vbox">
          <label text="双臂积木协作 · 环境预览" style="font-size: 16px; font-weight: bold;"/>
          <label text="橙色 A 臂 / 蓝色 B 臂 / 黄色临时桥墩"/>
          <label text="以下按钮切换预设状态，不是自主恢复算法。"/>
          <button text="① 正常布局" on-click="normalPreset"/>
          <button text="② A 臂受限，B 托住桥面" on-click="blockedPreset"/>
          <button text="③ 临时支撑到位，B 可以安装" on-click="supportPreset"/>
          <group layout="hbox" flat="true"><button text="全景视角" on-click="overview"/><button text="近景视角" on-click="detail"/></group>
          <label id="10" text="" wordwrap="true" style="max-width: 340px; color: #234366;"/>
          <label text="按住鼠标左键旋转视角，滚轮缩放。"/>
          <label text="当前版本：运动学场景；接触动力学与规划待接入。"/>
        </ui>]]
        ui = simUI.create(xml)
    end
    return true
end

function setPreset(mode)
    if not cfg then initialize() end
    if sim.getSimulationState() ~= sim.simulation_stopped then
        if ui then simUI.setLabelText(ui, 10, '请先停止仿真，再切换预设状态。') end
        return false
    end
    for _, side in ipairs({'A','B'}) do
        for i, alias in ipairs(cfg.arm_joint_aliases[side]) do
            local j = byAlias(alias)
            sim.setJointInterval(j, true, {-math.pi, 2*math.pi})
            local pose = side..'_home'
            if mode == 1 then pose = side == 'A' and 'A_hold' or 'B_support' end
            if mode == 2 and side == 'B' then pose = 'B_install' end
            sim.setJointPosition(j, cfg.poses[pose][i])
        end
    end
    local connector = byAlias('RedConnector')
    sim.setObjectParent(connector, root, true)
    sim.setObjectQuaternion(connector, {0,0,0,1}, sim.handle_world)
    if mode == 1 then
        sim.setObjectPosition(connector, {-0.285,-0.12,0.999}, sim.handle_world)
    elseif mode == 2 then
        sim.setObjectPosition(connector, {0.203,0.22,cfg.bridge_bottom+0.079}, sim.handle_world)
    else
        sim.setObjectPosition(connector, cfg.connector_initial, sim.handle_world)
    end
    sim.setObjectPosition(byAlias('TemporaryPier'), mode == 2 and cfg.prop_support or cfg.prop_park, sim.handle_world)
    sim.setObjectInt32Param(byAlias('FaultIndicator'), sim.objintparam_visibility_layer, mode == 0 and 0 or 1)
    sim.setObjectInt32Param(byAlias('B_SupportShoe'), sim.objintparam_visibility_layer, mode == 1 and 1 or 0)
    for name,h in pairs(objects) do
        if string.match(name, '^B_Finger') then
            sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, mode == 1 and 0 or 1)
        end
    end
    if mode ~= 0 then
        local j = byAlias('A_joint_3')
        local q = sim.getJointPosition(j)
        -- Test event only: keep current pose legal and narrow future elbow motion.
        sim.setJointInterval(j, false, {q-math.rad(12), math.rad(24)})
    end
    local text = ({
        [0]='正常布局：左右料盘、共享暂存区、城堡连桥和临时桥墩。',
        [1]='预设受限状态：A 拿连接砖，B 在桥下保持支撑。A 肘关节范围被缩小。',
        [2]='预设支撑状态：黄色桥墩已到位，B 位于安装端点。非自动规划结果。'
    })[mode]
    if ui then simUI.setLabelText(ui, 10, text) end
    cfg.preset=mode
    sim.writeCustomTableData(root, 'APEX_MR_Plus', cfg)
    return true
end

function normalPreset() setPreset(0) end
function blockedPreset() setPreset(1) end
function supportPreset() setPreset(2) end
function overview()
    sim.setObjectPose(sim.getObject('/DefaultCamera'), sim.getObjectPose(byAlias('OverviewRGB'), sim.handle_world), sim.handle_world)
end
function detail()
    sim.setObjectPose(sim.getObject('/DefaultCamera'), sim.getObjectPose(byAlias('DetailRGB'), sim.handle_world), sim.handle_world)
end
function getPreviewState()
    return {preset=cfg.preset, stage='environment_preview', planner_connected=false}
end
function sysCall_init() initialize() end
function sysCall_cleanup() if ui then simUI.destroy(ui) ui=nil end end
