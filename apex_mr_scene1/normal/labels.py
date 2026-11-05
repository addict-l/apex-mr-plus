"""Embed the normal-mode provenance labels in its runnable scene."""
from pathlib import Path
import sys
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from label_mesh import make_label


def apply_normal_labels(sim,root):
    font_path=r'C:\Windows\Fonts\msyhbd.ttc'
    rows=[('Title','normal_title.png',(21,40,65),(255,255,255),
           ['APEX MR PLUS','正常模式 · ILP / RRT-Connect / TPG'],[130,70]),
          ('Legend','normal_legend.png',(245,248,252),(30,48,70),
           ['动态 UR5 · 刚体积木 · 中央调度执行','刚性抓取 / 扣接近似；PadFT 实测支撑力'],[80,66])]
    for alias,filename,bg,fg,text,sizes in rows:
        image=Image.new('RGB',(2048,384),bg);draw=ImageDraw.Draw(image)
        for words,y,size in zip(text,[26,232],sizes):
            font=ImageFont.truetype(font_path,size)
            while draw.textlength(words,font=font)>1900:
                size-=2;font=ImageFont.truetype(font_path,size)
            width=draw.textlength(words,font=font)
            draw.text(((2048-width)/2,y),words,font=font,fill=fg)
        path=ROOT/'assets'/filename;image.save(path)
        old=sim.getObject('/APEX_MR_Plus/'+alias);pos=sim.getObjectPosition(old,-1)
        sim.removeObjects([old])
        make_label(sim,root,alias,path,pos,[1.08,.2025],True)
