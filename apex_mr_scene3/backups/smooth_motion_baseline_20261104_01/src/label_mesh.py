"""Textured planar labels with explicit UVs, independent of physical size."""
import math
from pathlib import Path
from PIL import Image


def make_label(sim, root, name, png, pos, size, vertical=False):
    im=Image.open(png).convert('RGB')
    w,h=size
    vertices=[-w/2,-h/2,0, w/2,-h/2,0, w/2,h/2,0, -w/2,h/2,0]
    indices=[0,1,2,0,2,3]
    uv=[0,0,1,0,1,1, 0,0,1,1,0,1]
    handle=sim.createShape(8|64,0,vertices,indices,None,uv,im.tobytes(),list(im.size))
    sim.setObjectAlias(handle,name)
    sim.setObjectParent(handle,root,True)
    sim.setObjectPosition(handle,-1,list(pos))
    sim.setObjectOrientation(handle,-1,[math.pi/2 if vertical else 0,0,0])
    sim.setShapeColor(handle,None,sim.colorcomponent_ambient_diffuse,[1,1,1])
    sim.setObjectInt32Param(handle,sim.shapeintparam_static,1)
    sim.setObjectInt32Param(handle,sim.shapeintparam_respondable,0)
    sim.setObjectSpecialProperty(handle,sim.objectspecialproperty_renderable)
    return handle


def replace_labels(sim, root, assets):
    specs={
      'Title':('title.png',[1.08,.2025],True),
      'Legend':('legend.png',[1.08,.2025],True),
      'A_RobotLabel':('label_A.png',[.31,.0775],False),
      'B_RobotLabel':('label_B.png',[.31,.0775],False),
      'A_TrayLabel':('label_supply_A.png',[.27,.0675],False),
      'B_TrayLabel':('label_supply_B.png',[.27,.0675],False),
      'CastleLabel':('label_goal.png',[.41,.1025],False),
      'BufferLabel':('label_buffer.png',[.28,.07],False),
      'PierLabel':('label_prop.png',[.20,.05],False),
      'FaultIndicator':('label_fault.png',[.28,.07],True),
    }
    for alias,(png,size,vertical) in specs.items():
        old=sim.getObject('/APEX_MR_Plus/'+alias)
        pos=sim.getObjectPosition(old,-1)
        visibility=sim.getObjectInt32Param(old,sim.objintparam_visibility_layer)
        sim.removeObjects([old])
        new=make_label(sim,root,alias,Path(assets)/png,pos,size,vertical)
        sim.setObjectInt32Param(new,sim.objintparam_visibility_layer,visibility)
