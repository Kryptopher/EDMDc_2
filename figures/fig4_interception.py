# Release copy of paper_acc_revision/final_figures/plot_interception_current_cv_simplified.py; only input/output paths and
# writes into the research repository were changed. Plotting code is unchanged.
"""Simplified three-family Current/CV interception figure from frozen traces only."""
from __future__ import annotations
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
INDEX=ROOT/'results'/'interception'/'episodes.json'
TRACES=ROOT/'results'/'figure_inputs'/'interception'
CONTROLLERS=ROOT/'configs'/'interception_controllers.json'
SEEDS=[('accelerating',3103001001),('helix',3103002006),('weaving',3103003001)]
# The old EPS uses red target, blue linear MPC, and green EDMDc-MPC.
COL={'Target':'#C62828','Linear-A':'#0064A7','EDMDc-B':'#00802B'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.9,'axes.titlesize':9.5,'xtick.labelsize':7.8,'ytick.labelsize':7.8,'legend.fontsize':8.8,'axes.linewidth':.7,'lines.linewidth':1.15,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.bbox':'tight'})

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(ok,msg):
    if not ok:raise AssertionError(msg)

def continuation_color(color):
    # Explicitly lighten RGB because EPS cannot represent alpha transparency.
    return tuple(.60*np.asarray(to_rgb(color))+.40)

def qualify(states,reference,time):
    distance=np.linalg.norm(states[:,:3]-reference[:,:3],axis=1)
    relative_speed=np.linalg.norm(states[:,3:6]-reference[:,3:6],axis=1)
    eligible=(distance<=.75)&(relative_speed<=3.0)
    consecutive=0;qual_index=None
    for k,yes in enumerate(eligible):
        consecutive=consecutive+1 if yes else 0
        if consecutive==21 and qual_index is None:qual_index=k
    return distance,relative_speed,qual_index

def load_cases():
    configs=read(CONTROLLERS)['configurations']
    verify(configs['A']['config']['horizon_steps']==50 and configs['A']['config']['block_steps']==10,'Linear-A config')
    verify(configs['B']['config']['horizon_steps']==100 and configs['B']['config']['block_steps']==5,'EDMDc-B config')
    allrows=read(INDEX);out=[]
    for family,seed in SEEDS:
        rows={row['cell']:row for row in allrows if row['seed']==seed and row['family']==family and row['mode']=='cv' and row['cell'] in ('Linear-A','EDMDc-B')}
        verify(set(rows)=={'Linear-A','EDMDc-B'},f'missing paired current/CV episodes {seed}')
        a,b=rows['Linear-A'],rows['EDMDc-B']
        verify(a['case_id']==b['case_id'] and a['reference_sha256']==b['reference_sha256'],f'paired identity {seed}')
        pair={};truth=None
        for label,row in (('Linear-A',a),('EDMDc-B',b)):
            path=TRACES/f'{label}_cv_{seed}.npz'
            with np.load(path) as z:states=z['states'].copy();reference=z['reference'].copy();time=z['time'].copy()
            verify(len(states)==len(reference)==len(time)==2001,'trace length')
            verify(truth is None or np.array_equal(reference,truth),'true target mismatch');truth=reference
            verify(np.array_equal(time,np.arange(2001)*.01),f'time axis mismatch {seed}')
            distance,speed,qi=qualify(states,reference,time)
            verify((qi is not None)==row['captured'],f'capture flag mismatch {seed}/{label}')
            verify((None if qi is None else float(time[qi]))==row['qualified_time_s'],f'qualification time mismatch {seed}/{label}')
            pair[label]={'row':row,'path':path,'states':states,'time':time,'distance':distance,'relative_speed':speed,'qualification_index':qi}
        out.append({'family':family,'seed':seed,'truth':truth,'arms':pair})
    return out

def format3d(ax):
    ax.view_init(elev=25,azim=-120)
    ax.set_xlabel('x [m]',labelpad=-5);ax.set_ylabel('y [m]',labelpad=-5);ax.set_zlabel('z [m]',labelpad=-10)
    ax.tick_params(labelsize=7.2,pad=-2)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=True;axis.pane.set_facecolor((.97,.97,.97,1))
        axis._axinfo['grid']['linewidth']=.35
        axis._axinfo['grid']['color']=(.84,.84,.84,1)

def main():
    cases=load_cases()
    latest_linear=max(case['arms']['Linear-A']['row']['qualified_time_s'] for case in cases if case['arms']['Linear-A']['row']['qualified_time_s'] is not None)
    end_s=min(20,int(np.ceil(latest_linear+2.0)))
    visible=end_s*100+1
    verify(all(v['qualification_index'] is None or v['qualification_index']<visible for case in cases for v in case['arms'].values()),'qualified capture outside visible window')
    # Match the old EPS: target families are columns, with trajectories above
    # separation. Relative speed adds the third row.
    # A slightly taller first grid row prevents Matplotlib's 3-D aspect logic
    # from narrowing the trajectory axes relative to the two rows below.
    fig=plt.figure(figsize=(7.12,6.65));grid=fig.add_gridspec(3,3,height_ratios=[1.32,1,1],left=.065,right=.99,bottom=.07,top=.92,wspace=.25,hspace=.28)
    all_distance=[v['distance'][:visible] for case in cases for v in case['arms'].values()]
    all_speed=[v['relative_speed'][:visible] for case in cases for v in case['arms'].values()]
    separation_top=max(float(v.max()) for v in all_distance)*1.04
    speed_top=max(float(v.max()) for v in all_speed)*1.04
    axes3=[];columns=[];positions=[]
    for i,case in enumerate(cases):
        ax3=fig.add_subplot(grid[0,i],projection='3d');sep=fig.add_subplot(grid[1,i]);vel=fig.add_subplot(grid[2,i]);axes3.append(ax3);columns.append((ax3,sep,vel))
        truth=case['truth'];t=np.arange(visible)*.01
        ax3.plot(*truth[:visible,:3].T,color=COL['Target'],ls='-',lw=1.15)
        ax3.scatter(*truth[0,:3],facecolors='white',edgecolors=COL['Target'],s=25,linewidths=1.05,depthshade=False)
        first=True
        for label in ('Linear-A','EDMDc-B'):
            item=case['arms'][label];states=item['states'];qi=item['qualification_index'];color=COL[label]
            before=visible if qi is None else qi+1
            ax3.plot(*states[:before,:3].T,color=color,lw=1.15,ls='-')
            if qi is not None and qi<visible-1:
                ax3.plot(*states[qi:visible,:3].T,color=continuation_color(color),lw=.85,ls=(0,(3,2)))
            if first:
                ax3.scatter(*states[0,:3],facecolors='white',edgecolors='#555555',s=25,linewidths=1.05,depthshade=False)
                first=False
            if qi is not None:ax3.scatter(*states[qi,:3],color=color,marker='*',s=72,linewidths=.45,edgecolors='white',depthshade=False)
            for ax,y in ((sep,item['distance']),(vel,item['relative_speed'])):
                ax.plot(t[:before],y[:before],color=color,lw=1.15,ls='-')
                if qi is not None and qi<visible-1:
                    ax.plot(t[qi:],y[qi:visible],color=continuation_color(color),lw=.85,ls=(0,(3,2)))
                if qi is not None:
                    ax.axvline(t[qi],color=continuation_color(color),ls=':',lw=.65,zorder=1)
                    ax.plot(t[qi],y[qi],marker='*',color=color,ms=7.2,markeredgecolor='white',markeredgewidth=.45,zorder=5)
            positions.append(states[:visible,:3]);
        positions.append(truth[:visible,:3])
        ax3.set_title(case['family'].capitalize(),pad=0);format3d(ax3)
        for ax,threshold,upper,ylabel in ((sep,.75,separation_top,'Separation [m]'),(vel,3.,speed_top,'Relative speed [m/s]')):
            ax.axhline(threshold,color=COL['Target'],ls=(0,(4,3)),lw=.8)
            ax.set_xlim(0,end_s);ax.set_ylim(0,upper)
            if i==0:ax.set_ylabel(ylabel)
            if ax is vel:ax.set_xlabel('Time [s]')
            ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.tick_params(direction='out',length=2.5)
    # Common limits allow direct geometric comparison between target families.
    xyz=np.concatenate(positions);lo=xyz.min(axis=0);hi=xyz.max(axis=0)
    for ax in axes3:
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(np.maximum(hi-lo,.01),zoom=1.16)
    fig.canvas.draw()
    # Shift only the 3-D row left within the finished grid; widths and data stay fixed.
    trajectory_shift=-.022
    for ax in axes3:
        box=ax.get_position()
        ax.set_position((box.x0+trajectory_shift,box.y0,box.width,box.height))
    fig.canvas.draw()
    widths=[ax.get_position().width for column in columns for ax in column]
    verify(max(widths)-min(widths)<1e-10,'unequal subplot widths')
    top_heights=[ax.get_position().height for ax in axes3]
    verify(max(top_heights)-min(top_heights)<1e-10,'unequal 3-D bounding-box heights')
    for column in columns:
        centers=[(ax.get_position().x0+ax.get_position().x1)/2 for ax in column]
        verify(abs(centers[0]-centers[1]-trajectory_shift)<1e-10 and abs(centers[1]-centers[2])<1e-10,'unexpected 3-D row shift')
    column_centers=[(column[0].get_position().x0+column[0].get_position().x1)/2 for column in columns]
    verify(abs((column_centers[1]-column_centers[0])-(column_centers[2]-column_centers[1]))<1e-10,'unequal column spacing')
    lines_=[Line2D([0],[0],color=COL['Target'],ls='-',lw=1.15,label='Target'),Line2D([0],[0],color=COL['Linear-A'],lw=1.15,label='Linear-A'),Line2D([0],[0],color=COL['EDMDc-B'],lw=1.15,label='EDMDc-B')]
    # Neutral symbol keys apply to both controllers; plotted markers retain
    # their target/controller colors. Matplotlib fills the 3-column legend
    # column-major, giving controller curves on row 1 and symbol meanings on row 2.
    initial_key=Line2D([0],[0],ls='none',marker='o',ms=5.5,mfc='white',mec='#555555',mew=1.05,label='Open circle: initial position')
    capture_key=Line2D([0],[0],ls='none',marker='*',ms=8.5,color='#555555',mec='white',mew=.45,label='Star: qualified capture')
    handles=[lines_[0],initial_key,lines_[1],capture_key,lines_[2]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.03),ncol=3,frameon=False,columnspacing=1.6,handletextpad=.5)
    for ext in ('pdf','svg','eps','png'):fig.savefig(OUT/f'fig4_interception.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
    print('Seeds:',', '.join(str(s) for _,s in SEEDS))
    for case in cases:
        print(case['family'],case['seed'],'Linear-A qualification',case['arms']['Linear-A']['row']['qualified_time_s'],'EDMDc-B qualification',case['arms']['EDMDc-B']['row']['qualified_time_s'])
    print('Thresholds shown: 0.75 m separation and 3 m/s relative speed; markers/vertical lines are 21-sample dwell-end qualification only.')
    print('Common plot window:',end_s,'s; solid through qualified capture, lighter dashed continuation afterward.')
    print('Simulations rerun: no.')
    for ext in ('pdf','svg','eps','png'):print(OUT/f'fig4_interception.{ext}')

if __name__=='__main__':main()
