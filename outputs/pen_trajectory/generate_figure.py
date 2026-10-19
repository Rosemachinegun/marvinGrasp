"""Offline, reproducible academic figure using the repository's current planners."""
from pathlib import Path
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grasp_core.execution.config import GRASP_CONFIG, HOME_CONFIG
from grasp_core.core.types.target_pose import TargetObjectPose
from grasp_core.core.math.pose import ik_home_wrist_orientation_quat, select_ik_hand
from grasp_core.planning.grasp.planner import UnifiedGraspPlanner
from grasp_core.planning.trajectory.planner import plan_trajectory
from grasp_core.planning.trajectory.constraints import effective_trajectory_step_limits

OUT = Path(__file__).resolve().parent
args = GRASP_CONFIG.controlled_args(argparse.Namespace(tool_template_path=ROOT/'grasp_core/resources/tool.yaml', use_tool_pick_template=True))
obj = np.array([.385, -.154, .670])
hand = select_ik_hand(obj, 'auto')
start = np.array(HOME_CONFIG.position(hand))
args = HOME_CONFIG.apply_runtime_args(args, hand)
q0 = ik_home_wrist_orientation_quat(args, hand)
pose = np.eye(4)
pose[:3, 3] = obj
# Illustrative slender geometry and horizontal long axis; dimensions are not measurements.
target = TargetObjectPose('pen', 'hypothetical_base', pose.copy(), pose, np.array([.14, .012, .012]))
planner = UnifiedGraspPlanner()
templates = planner.load_templates(args)
plan = planner.build_template_plan(target, templates['pen'][hand], start, q0, args, hand)
limits = effective_trajectory_step_limits(args)
traj = plan_trajectory(start, q0, plan.approach_waypoints + plan.final_approach_waypoints,
    max_step_m=limits[0], max_step_deg=limits[1], min_steps=1,
    final_slowdown_ratio=GRASP_CONFIG.final_approach_slowdown_ratio, smoothing_passes=2)
p = np.array([v[0] for v in traj.samples])
q = np.array([v[1] for v in traj.samples])
s = np.linspace(0, 1, len(p))
join = traj.segment_steps[0]-1
pre = plan.approach_waypoints[-1][0]
grip = plan.grip_position
assert np.allclose(p[0], start) and np.allclose(p[-1], grip)
assert np.allclose(p[join], pre)
assert np.allclose(pre, [.385, -.154, .770])
assert np.allclose(grip, [.385, -.154, .700])
assert np.isfinite(p).all()
np.savetxt(OUT/'pen_trajectory_samples.csv', np.column_stack([np.arange(len(p)), s, p, q]), delimiter=',',
    header='sample_index,normalized_sample_progress,x_m,y_m,z_m,qx,qy,qz,qw', comments='')
meta = dict(object_xyz_m=obj.tolist(), start_xyz_m=start.tolist(), pregrasp_xyz_m=pre.tolist(),
    grasp_xyz_m=grip.tolist(), start_quaternion_xyzw=list(q0), hand=hand, sample_count=len(p),
    segment_steps=traj.segment_steps, path_length_m=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),
    max_sample_step_m=float(np.linalg.norm(np.diff(p,axis=0),axis=1).max()), effective_step_limits=list(limits),
    assumptions=['Base-frame coordinates; object z=0.670 is already the planning input.',
    'Start at configured right Home pose, not measured hardware feedback.',
    'Horizontal pen long axis along base +X; illustrative size 0.14 x 0.012 x 0.012 m.',
    'Planning samples only; no IK, collision checking or hardware execution performed.',
    'Normalized sample index is not measured time; no post-grasp lift in this grasp routine.'])
(OUT/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.titlesize':13,
 'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
 'svg.fonttype':'path','figure.facecolor':'white','axes.edgecolor':'#8996a5','text.color':'#182a3c',
 'axes.labelcolor':'#24384b','xtick.color':'#536579','ytick.color':'#536579'})
blue, orange, gray = '#176B9A', '#DC7D28', '#9CA6AF'
fig=plt.figure(figsize=(16,9))
gs=fig.add_gridspec(3,2,left=.045,right=.96,bottom=.19,top=.80,width_ratios=[1.23,1],hspace=.32,wspace=.18)
ax=fig.add_subplot(gs[:,0], projection='3d')
ax.plot(*p[:join+1].T,color=blue,lw=3,label='接近预抓取点')
ax.plot(*p[join:].T,color=orange,lw=3,label='末端下降抓取')
ax.plot(*np.array([start,pre,grip]).T,'--',color=gray,lw=1.2,alpha=.8,label='原始航点连线')
for pt, mark, color, label in [(start,'o',blue,'S  起点'),(pre,'s',blue,'P  预抓取'),(grip,'D',orange,'G  抓取')]:
 ax.scatter(*pt,s=55,c=color,marker=mark,edgecolors='white',linewidths=1,depthshade=False,zorder=6)
 ax.text(pt[0]-.004,pt[1]+.008,pt[2]+.006,label,fontsize=10)
ax.scatter(*obj,marker='*',s=150,color='#B74459',depthshade=False,label='pen 中心')
ax.plot([obj[0],grip[0]],[obj[1],grip[1]],[obj[2],grip[2]],':',color='#B74459',lw=1.5)
ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
ax.set_xlim(.235,.408); ax.set_ylim(-.265,-.135); ax.set_zlim(.65,.85)
ax.set_box_aspect((.173,.130,.200))
ax.view_init(elev=24,azim=-57)
for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
 axis.set_major_locator(MaxNLocator(5)); axis.set_major_formatter(FormatStrFormatter('%.2f'))
 axis.pane.fill=False
 axis._axinfo['grid'].update(color='#DDE3E9',linewidth=.6)
ax.set_title('(a)  末端笛卡尔空间轨迹',loc='left',pad=14,fontweight='bold')
ax.legend(loc='upper left',bbox_to_anchor=(-.015,.94),frameon=False,fontsize=9)
for k in [int(join*.48),join+max((len(p)-join)//2,1)]:
 d=p[min(k+5,len(p)-1)]-p[k]
 ax.quiver(*p[k],*d,normalize=True,length=.015,color=blue if k<join else orange,arrow_length_ratio=.45)
for i, coord in enumerate('XYZ'):
 a=fig.add_subplot(gs[i,1])
 a.axvspan(s[join],1,color=orange,alpha=.065,lw=0)
 a.plot(s[:join+1],p[:join+1,i],color=blue,lw=2.4)
 a.plot(s[join:],p[join:,i],color=orange,lw=2.4)
 a.axvline(s[join],color=gray,lw=1,ls='--')
 a.scatter([0,s[join],1],[p[0,i],p[join,i],p[-1,i]],c=[blue,blue,orange],s=25,zorder=4)
 a.set_xlim(-.025,1.025); a.set_ylabel(f'{coord} (m)');a.grid(axis='y',color='#E4E9EE',lw=.7)
 a.yaxis.set_major_locator(MaxNLocator(4));a.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
 a.margins(y=.22)
 if i==0:
  a.set_title('(b)  各坐标分量的插值曲线',loc='left',pad=14,fontweight='bold')
  a.text(.03,.86,'接近',transform=a.transAxes,color=blue)
  a.text(.94,.13,'抓取',transform=a.transAxes,color=orange,ha='right')
 if i<2:a.tick_params(labelbottom=False)
 else:a.set_xlabel('归一化采样进度  u = i / (N − 1)')
 a.annotate(f'{p[-1,i]:.3f}',(1,p[-1,i]),xytext=(-9,10),textcoords='offset points',ha='right',color=orange,fontsize=10)
fig.text(.05,.94,'pen 抓取过程的末端轨迹插值',fontsize=23,fontweight='bold')
fig.text(.05,.892,'目标中心 (0.385, −0.154, 0.670) m   |   右臂   |   基坐标系   |   基于当前规划代码的离线计算',fontsize=12,color='#536579')
fig.text(.05,.126,'S  (0.250, −0.250, 0.830)     →     P  (0.385, −0.154, 0.770)     →     G  (0.385, −0.154, 0.700) m',fontsize=12,fontweight='bold')
fig.text(.05,.079,'方法：分段三次 Bézier 插值 + 两次采样链平滑；姿态采用四元数 SLERP。抓取结束于 G，夹爪闭合。',fontsize=10.5,color='#536579')
fig.text(.05,.044,'假设：从配置 Home 位姿出发，pen 水平且长轴沿 +X。图示为末端目标轨迹，非关节轨迹或实测结果；横轴不代表实测时间。',fontsize=10,color='#536579')
for ext in ['png','svg']:
 fig.savefig(OUT/f'pen_grasp_trajectory.{ext}',dpi=300,facecolor='white')
fig.savefig(OUT/'preview.png',dpi=110)
print(json.dumps(meta,indent=2))
