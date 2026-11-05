"""Qualify skill modes and compile checked direct paths or genuine RRT paths."""
from copy import deepcopy
from pathlib import Path
import hashlib,inspect,json,math,time
import numpy as np
from scipy.spatial.transform import Rotation as R
from .motion import MotionPlanner
from .allocation import solve_allocation

ROOT=Path(__file__).resolve().parents[1]
DOWN={'A':[1.,0,0,0],'B':[0,1.,0,0]}
FORWARD=R.from_euler('x',-math.pi/2).as_quat().tolist()
SIDE=(R.from_euler('y',-math.pi/2)*R.from_euler('z',math.pi/2)).as_quat().tolist()

def save(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def transform(pose,offset,quat):
    rot=R.from_quat(pose[3:]);return (np.array(pose[:3])+rot.apply(offset)).tolist()+(rot*R.from_quat(quat)).as_quat().tolist()

def grasp(part,arm,kind=None):
    if part=='TemporarySupport':return [.032,0,.145],R.from_euler('y',-math.pi/2).as_quat().tolist()
    if part=='K1' and kind=='install':return [.058,0,0],SIDE
    if part=='Gate':return [0,-.006,0],FORWARD
    return [0,0,{'K1':.030,'K2':.030}[part]],DOWN[arm]

def binding(part,offset,quat):
    inv=R.from_quat(quat).inv()
    return {'part':part,'name':part,'offset':inv.apply(-np.asarray(offset)).tolist(),'quaternion':inv.as_quat().tolist()}

class CompileFailure(RuntimeError):
    def __init__(self,label,evidence):
        self.label=label;self.evidence=evidence
        super().__init__(label+': '+str(evidence.get('problem') or evidence.get('error') or evidence))

class Compiler:
    def __init__(self,sim,planner,cfg,outdir):
        self.sim,self.p,self.cfg,self.out=sim,planner,cfg,Path(outdir)
        self.home={s:list(a['home_q']) for s,a in cfg['arms'].items()}
        self.source={k:(v if len(v)==7 else v+[0,0,0,1]) for k,v in cfg['initial'].items()}
        self.goals={k:(v if len(v)==7 else v+[0,0,0,1]) for k,v in cfg['goal'].items()}
        self.buffer={'K1':[0,-.36,.806,0,0,0,1],'K2':[0,-.36,.808,0,0,0,1]}
        self.calls=[];self.qual=[];self.cache=self.out/'rrt_paths';self.cache.mkdir(exist_ok=True)
        self.extension=float(cfg.get('runtime',{}).get('support_extension_m',.008))
        if not 0<self.extension<=.012:raise ValueError('Support extension must fit the physical 12mm actuator range')
        self.ik_cache=self.out/'ik_solutions';self.ik_cache.mkdir(exist_ok=True)
        self.sig=hashlib.sha256((ROOT/'scenes/10_Scene3_Dynamic.ttt').read_bytes()+(ROOT/'apex/motion_backend.lua').read_bytes()).hexdigest()
        # A completed historical reference supplies initial guesses only.
        # Changed bodies, contacts and payloads still require current-scene IK.
        self.reference_ik_seeds={}
        try:
            prior=json.loads((self.out/'reference_tasks.json').read_text(encoding='utf-8'))
            if isinstance(prior,list):
                for task in prior:
                    if not isinstance(task,dict):continue
                    q=task.get('q_end')
                    if q is None and isinstance(task.get('q_path'),list) and task['q_path']:q=task['q_path'][-1]
                    if (isinstance(task.get('id'),str) and task.get('arm') in ('A','B') and
                        isinstance(q,list) and len(q)==6 and all(isinstance(v,(int,float)) and
                        not isinstance(v,bool) and math.isfinite(v) for v in q)):
                        self.reference_ik_seeds[task['id']]={'arm':task['arm'],'q':list(q)}
        except (OSError,ValueError,TypeError):
            pass  # Missing/corrupt prior reference is not planning evidence.
        self.reference_paths={}
        for file,source in (
            (ROOT/'backups/smooth_motion_baseline_20261104_01/results/apex/reference_tasks.json','baseline_reference'),
            (self.out/'reference_tasks.json','previous_reference')):
            try:
                reference=json.loads(file.read_text(encoding='utf-8'))
                if not isinstance(reference,list):continue
                for task in reference:
                    if not isinstance(task,dict) or task.get('arm') not in ('A','B') or not isinstance(task.get('id'),str):continue
                    path=task.get('q_path')
                    if not isinstance(path,list) or len(path)<2:continue
                    if not all(isinstance(q,list) and len(q)==6 and all(
                        isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in q) for q in path):continue
                    self.reference_paths.setdefault((task['arm'],task['id']),[]).append(
                        {'path':deepcopy(path),'source':source+':'+task['id']})
            except (OSError,ValueError,TypeError):
                pass  # Historical geometry is only a candidate, never a certificate.
        self.reset()

    def reset(self):
        self.q=deepcopy(self.home);self.placed=deepcopy(self.source);self.payloads={'A':None,'B':None}
        self.art={'SupportLift':0.};self.support_contacts=[['A_SupportShoe','Bridge']]
        self.contact_holds={}
        self.tasks=[];self.last={'A':None,'B':None}

    def tool_contacts(self,arm,part):
        return [[self.sim.getObjectAlias(h,1),part] for h in self.cfg['arms'][arm]['shapes']
                if any(x in self.sim.getObjectAlias(h) for x in ('Palm','Finger','SupportShoe'))]

    def resting_contacts(self,part):
        contacts=[[part,self.sim.getObjectAlias(h,1)] for h in [self.cfg['table'],*self.cfg['obstacles']]
                  if not self.sim.getObjectAlias(h).startswith('Pocket')]
        if part=='K1':contacts.append(['K1','Bridge'])
        if part=='K2':contacts.append(['K2','Bridge'])
        if part=='TemporarySupport':contacts.append(['SupportContactPad','Bridge'])
        return contacts

    def contact_context(self,extra=()):
        """Exact contact pairs, including a released gripper waiting in place."""
        groups=[self.support_contacts,*self.contact_holds.values(),list(extra)]
        pairs=[];seen=set()
        for group in groups:
            for pair in group:
                if len(pair)!=2:raise ValueError('Contact exclusions must be exact object pairs')
                key=tuple(sorted(map(str,pair)))
                if key not in seen:seen.add(key);pairs.append(deepcopy(list(pair)))
        return pairs

    def clear_contact_hold(self,label,arm,kind='tool'):
        """Clear after a complete retreat, once, not on every path fragment.

        An explicit atomic wait keeps the contact permission while a geometric
        TPG dependency pauses an intermediate retreat fragment. It then clears
        only this arm's temporary permission at the measured retreat endpoint.
        """
        key=kind+'_'+arm
        if key not in self.contact_holds:return
        self.skill(label+'_contact_clear',arm,'wait',dwell_s=.02,
                   effects={'support_contacts':{key:None}})
        self.contact_holds.pop(key,None)

    def options(self,arm,contacts=(),**kw):
        return {'other_q':self.q['B' if arm=='A' else 'A'],
                'other_payload':self.payloads['B' if arm=='A' else 'A'] or {},
                'contact_exclusions':self.contact_context(contacts),'contact_tolerance':.002,
                'articulations':deepcopy(self.art),**kw}

    def ik(self,arm,tcp,payload=None,contacts=(),seed=None,label='IK'):
        hint=getattr(self,'reference_ik_seeds',{}).get(label)
        # Continue the live configuration branch before consulting historical
        # hints. Explicit goal-branch seeds remain authoritative at narrow
        # insertion approaches. A cached distant fallback must not preempt a
        # fresh solve from the current configuration.
        seeds=[]
        if seed is not None:seeds.append((seed,True,'explicit'))
        seeds.append((self.q[arm],True,'current'))
        if hint and hint['arm']==arm:seeds.append((hint['q'],False,'prior_reference'))
        seeds.extend([(self.cfg['free_q'][arm],False,'free'),
            (self.home[arm],False,'home'),
            (([-2.31061867,1.02772691,.67678082,-.13371371,-1.57079609,-.73982243]
              if arm=='A' else [-1.08916658,1.02772498,.67678477,-.13371583,-1.57079608,.48162979]),False,'default')])
        requests=[];seen=set()
        for qseed,is_preferred,source in seeds:
            request=deepcopy({'arm':arm,'tcp':tcp,'payload':payload,'placed':self.placed,
                'options':self.options(arm,contacts,seed=qseed,ik_time=.4,max_solutions=16)})
            # Preserve every numeric digit, quaternion component, object pose,
            # other-arm payload/configuration, contact pair, articulation and
            # solver option. Labels are not part of the geometric request.
            encoded=json.dumps(request,sort_keys=True,separators=(',',':'),allow_nan=False)
            key=hashlib.sha256((self.sig+'\n'+encoded).encode()).hexdigest()
            if key not in seen:
                seen.add(key);requests.append((key,request,self.ik_cache/(key+'.json'),is_preferred,source))
        def successful(value):
            q=value.get('q') if isinstance(value,dict) else None
            return (isinstance(value,dict) and value.get('ok') is True and isinstance(q,list) and len(q)==6 and
                    all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in q))
        def cached_result(request,file):
            if not file.is_file():return None
            try:
                entry=json.loads(file.read_text(encoding='utf-8'))
                if (entry.get('schema_version')==1 and entry.get('scene_backend_signature')==self.sig and
                    entry.get('request')==request and successful(entry.get('result'))):
                    return entry['result']
            except (OSError,ValueError,TypeError,AttributeError):
                pass  # A partial/corrupt cache is a miss, never IK evidence.
            return None
        def solve(request,file):
            value=self.p.solve_ik(arm,request['tcp'][:3],request['tcp'][3:],
                request['payload'],request['placed'],request['options'])
            if value.get('ok'):
                if not successful(value):raise CompileFailure(label,{'problem':'IK reported success without a finite six-joint solution'})
                save(file,{'schema_version':1,'scene_backend_signature':self.sig,'request':request,'result':value})
            return value
        cached=False;result=None;chosen_key=None;chosen_source=None
        # Evaluate in seed priority order, using only an exact current-scene
        # cache entry for that request. A cache hit from a later, distant seed
        # must not outrank an earlier branch which has not yet been refined.
        for key,request,file,is_preferred,source in requests:
            result=cached_result(request,file);cached=result is not None
            if result is None:result=solve(request,file)
            chosen_key,chosen_source=key,source
            if result.get('ok'):break
        self.calls.append({'type':'IK','label':label,'arm':arm,'tcp':tcp,'ok':result.get('ok',False),
                           'cached':cached,'cache_key':chosen_key,'seed_source':chosen_source,
                           'problem':result.get('problem'),'stats':result.get('stats')})
        if not result.get('ok'):raise CompileFailure(label,result)
        return list(result['q'])

    def candidate_paths(self,label,arm,end,max_endpoint_correction=.12):
        """Align nearby historical branches; every proposal needs fresh checks.

        A single 2pi shift per joint preserves the historical curve's winding.
        Only small endpoint changes are distributed along its arc length; a
        different IK branch is rejected instead of deforming it across space.
        """
        candidates=[];seen=set();start=self.q[arm]
        for record in getattr(self,'reference_paths',{}).get((arm,label),[]):
            path=record['path']
            shifts=[2*math.pi*math.floor((a-b)/(2*math.pi)+.5) for a,b in zip(start,path[0])]
            aligned=[[q[j]+shifts[j] for j in range(6)] for q in path]
            first=[a-b for a,b in zip(start,aligned[0])]
            last=[a-b for a,b in zip(end,aligned[-1])]
            correction=max(abs(x) for x in first+last)
            if correction>max_endpoint_correction:continue
            length=[0.]
            for a,b in zip(aligned,aligned[1:]):length.append(length[-1]+math.sqrt(sum((x-y)**2 for x,y in zip(a,b))))
            adjusted=[]
            for index,q in enumerate(aligned):
                fraction=length[index]/length[-1] if length[-1]>1e-12 else index/(len(aligned)-1)
                adjusted.append([q[j]+(1-fraction)*first[j]+fraction*last[j] for j in range(6)])
            adjusted[0]=list(start);adjusted[-1]=list(end)
            key=json.dumps(adjusted,separators=(',',':'),allow_nan=False)
            if key in seen:continue
            seen.add(key)
            candidates.append({'path':adjusted,'source':record['source'],
                               'max_endpoint_correction_rad':correction})
        return candidates[:4]

    def goal_windings(self,label,arm,end):
        """Bounded equivalent goals suggested by validated-plan history.

        A nearest IK representation need not be connected to the start in free
        space. Preserve known alternative revolutions, without inventing new
        Cartesian goals or treating endpoint validity as a path certificate.
        The backend remains authoritative for each joint's actual limits.
        """
        result=[{'goal':list(end),'source':'requested_goal'}]
        for record in getattr(self,'reference_paths',{}).get((arm,label),[]):
            path=record['path'];period=2*math.pi
            aligned=[path[-1][j]+period*math.floor((self.q[arm][j]-path[0][j])/period+.5) for j in range(6)]
            variant=[end[j]+period*math.floor((aligned[j]-end[j])/period+.5) for j in range(6)]
            if any(q < -period-1e-10 or q > period+1e-10 for q in variant):continue
            if any(max(abs(a-b) for a,b in zip(variant,old['goal']))<1e-8 for old in result):continue
            result.append({'goal':variant,'source':record['source']})
        return result[:4]

    def path(self,label,arm,end,payload=None,contacts=(),slow=False,dependencies=()):
        constrained=label.startswith(('support_slide_in_','extract_support_motion_','install_K1_slide_','A_exit_from_support_','close_gate_slide_','close_gate_retreat_')) or label in ('support_set_down','raise_unloaded_support','A_lower_from_support','install_K1_seat')
        base_options=self.options(arm,contacts,joint_bound_margin=.008 if constrained else 1.2)
        # OMPL's fraction is relative to the joint-space bounds. Tight local
        # bounds otherwise force ~0.0001rad sampling. The backend still caps
        # the final certificate step at .005rad (typically ~.001 here).
        if constrained:base_options['resolution']=.01
        variants=self.goal_windings(label,arm,end)
        attempts=[];best=None;best_cost=math.inf;failed=[]
        def query(variant,retry=False):
            goal=variant['goal'];options=deepcopy(base_options)
            if retry:
                if constrained:options['joint_bound_margin']=.025
                else:options.pop('joint_bound_margin',None)
            candidates=self.candidate_paths(label,arm,goal)
            if candidates:options['candidate_paths']=candidates
            budget=7. if retry else 3.
            req=dict(policy='bounded_reference_windings_v1',label=label,arm=arm,
                     start=self.q[arm],end=goal,payload=payload,placed=self.placed,options=options,solve_time=budget)
            key=hashlib.sha256((self.sig+json.dumps(req,sort_keys=True)).encode()).hexdigest()
            file=self.cache/(key+'.json')
            if file.exists():
                value=json.loads(file.read_text(encoding='utf-8'));cached=True
            else:
                print(json.dumps({'planning':label,'arm':arm,'goal_source':variant['source'],'retry':retry}),flush=True)
                value=self.p.plan(arm,self.q[arm],goal,payload,self.placed,solve_time=budget,options=options)
                cached=False
            evidence={'goal':goal,'goal_source':variant['source'],'retry':retry,'cached':cached,
                      'ok':value.get('ok',False),'stats':value.get('stats'),
                      'error':value.get('error'),'problem':value.get('problem')}
            if value.get('ok'):
                path=value.get('path',[])
                if (not path or max(abs(a-b) for a,b in zip(path[0],self.q[arm]))>1e-8 or
                    max(abs(a-b) for a,b in zip(path[-1],goal))>1e-8):
                    value={'ok':False,'error':'Returned path does not match resolved winding endpoints'}
                    evidence.update(ok=False,error=value['error'])
                else:
                    evidence['cost']=sum(math.sqrt(sum((a-b)**2 for a,b in zip(q,r))) for q,r in zip(path,path[1:]))
                    if not cached:save(file,value)
            attempts.append(evidence)
            return value,cached,options,evidence
        # Try alternate known windings before spending the larger retry budget
        # on an endpoint representation which may lie in a disconnected region.
        for variant in variants:
            lower_bound=math.sqrt(sum((a-b)**2 for a,b in zip(self.q[arm],variant['goal'])))
            if best is not None and lower_bound>=best_cost-1e-10:
                attempts.append({'goal':variant['goal'],'goal_source':variant['source'],
                                 'skipped':'endpoint_distance_not_better_than_best','lower_bound_cost':lower_bound})
                continue
            value,cached,options,evidence=query(variant)
            if value.get('ok'):
                if evidence['cost']<best_cost:best=(value,cached,options);best_cost=evidence['cost']
            else:failed.append(variant)
        if best is None:
            for variant in failed:
                value,cached,options,evidence=query(variant,True)
                if value.get('ok') and evidence['cost']<best_cost:
                    best=(value,cached,options);best_cost=evidence['cost']
        if best is None:
            raise CompileFailure(label,{'error':'No feasible path for bounded equivalent joint goals',
                                        'requested_goal':list(end),'goal_attempts':attempts,
                                        'problem':attempts[-1].get('problem') if attempts else None})
        result,cached,options=best
        result=deepcopy(result);resolved=list(result['path'][-1])
        result.setdefault('stats',{}).update(requested_goal=list(end),resolved_goal=resolved,
            winding_fallback=any(abs(a-b)>1e-8 for a,b in zip(end,resolved)),goal_attempts=attempts)
        algorithm=result.get('algorithm',result.get('stats',{}).get('algorithm','RRTConnect'))
        self.calls.append({'type':algorithm,'label':label,'ok':True,'cached':cached,'stats':result.get('stats')})
        metadata={'placed':deepcopy(self.placed),'contact_exclusions':options['contact_exclusions'],
                  'options':{'articulations':deepcopy(self.art)},
                  'planner':'simOMPL.RRTConnect' if algorithm=='RRTConnect' else algorithm,
                  'path_source':result.get('path_source',result.get('stats',{}).get('path_source','rrt_connect')),
                  'planning_stats':{'requested_goal':list(end),'resolved_goal':resolved,
                                    'winding_fallback':result['stats']['winding_fallback']}}
        if label.startswith(('install_K1','install_K2','close_gate')):metadata['assembly_force_guard']=True
        if slow:metadata.update(max_velocity_rad_s=.12,max_acceleration_rad_s2=.3)
        self.add(label,arm,'transit',result['path'],metadata,payload=payload,dependencies=dependencies)
        return result['path']

    def add(self,id,arm,kind,path,metadata=None,part=None,payload=None,dependencies=()):
        deps=list(dependencies)
        if self.last[arm]:deps.append(self.last[arm])
        task=dict(id=id,arm=arm,kind=kind,part=part,q_path=deepcopy(path),duration=float(max(.1,sum(np.linalg.norm(np.array(a)-b) for a,b in zip(path,path[1:]))/.4)),
                  dependencies=list(dict.fromkeys(deps)),reference_order=len(self.tasks),metadata=metadata or {},payload=deepcopy(payload))
        self.tasks.append(task);self.last[arm]=id;self.q[arm]=list(path[-1])
        save(self.out/'reference_tasks_partial.json',{'stage':'incomplete_reference_diagnostic','tasks':self.tasks})
        return id

    def skill(self,id,arm,action,part=None,target=None,dependencies=(),effects=None,**kw):
        # Contact RRT endpoints and the following atomic skill must use the
        # same exact tool/object and supported-object/surface permissions.
        # Grasp is still an UNHELD part until its completion event; release is
        # still a HELD part during its load/dwell interval.
        releases={'release_buffer','release_support','install','close_gate','return_support'}
        extra=list(kw.pop('contact_exclusions',[]))
        tool_pairs=[];resting_pairs=[]
        if part and (action=='grasp' or action in releases or action=='verify_support'):
            tool_pairs=self.tool_contacts(arm,part);resting_pairs=self.resting_contacts(part)
            extra+=tool_pairs+resting_pairs
        options=deepcopy(kw.pop('options',{}))
        extra+=options.pop('contact_exclusions',[])
        options={'articulations':deepcopy(self.art),**options}
        state_effects=deepcopy(effects or {})
        if action=='grasp':
            state_effects.setdefault('support_contacts',{}).update({'pickup_'+arm:resting_pairs,'tool_'+arm:None})
        elif action in releases:
            state_effects.setdefault('support_contacts',{}).update({'tool_'+arm:tool_pairs,'pickup_'+arm:None})
        md={'action':action,'placed':deepcopy(self.placed),'contact_exclusions':self.contact_context(extra),
            'contact_tolerance':.002,'options':options,'dwell_s':.3,**kw}
        if id.startswith(('install_K1','install_K2','close_gate')):md['assembly_force_guard']=True
        if target:md['target_pose']=list(target)
        if state_effects:md['effects']=state_effects
        result=self.add(id,arm,action,[self.q[arm],self.q[arm]],md,part,self.payloads[arm],dependencies)
        if action=='grasp':
            self.contact_holds['pickup_'+arm]=resting_pairs;self.contact_holds.pop('tool_'+arm,None)
        elif action in releases:
            self.contact_holds['tool_'+arm]=tool_pairs;self.contact_holds.pop('pickup_'+arm,None)
        return result

    def move_pose(self,label,arm,part,object_pose,grasp_spec,payload=None,contacts=(),slow=False,dependencies=()):
        tcp=transform(object_pose,*grasp_spec)
        q=self.ik(arm,tcp,payload,contacts,label=label)
        try:
            return self.path(label,arm,q,payload,contacts,slow,dependencies)
        except CompileFailure as original:
            branch_attempts=[];seen=[q]
            for record in getattr(self,'reference_paths',{}).get((arm,label),[])[:2]:
                try:
                    candidate=self.ik(arm,tcp,payload,contacts,seed=record['path'][-1],label=label+'_branch_fallback')
                    if any(max(abs(math.atan2(math.sin(a-b),math.cos(a-b))) for a,b in zip(candidate,old))<1e-5 for old in seen):continue
                    seen.append(candidate)
                    path=self.path(label,arm,candidate,payload,contacts,slow,dependencies)
                    self.tasks[-1]['metadata']['planning_stats']['ik_branch_fallback']=record['source']
                    self.calls[-1]['stats']['ik_branch_fallback']=record['source']
                    return path
                except CompileFailure as failure:
                    branch_attempts.append({'source':record['source'],'evidence':failure.evidence})
            original.evidence['ik_branch_fallback_attempts']=branch_attempts
            raise original

    def slide_pose(self,label,arm,part,start,end,spec,payload,contacts=()):
        # Constrain each short approach to a nearby IK branch. Every segment
        # is still collision checked by the common geometry oracle, using a
        # checked direct segment or RRTConnect. No Cartesian waypoint is
        # converted directly into an unchecked joint trajectory.
        steps=max(1,math.ceil(np.linalg.norm(np.asarray(end[:3])-start[:3])/.020))
        for index in range(1,steps+1):
            pose=list(end)
            pose[:3]=(np.asarray(start[:3])+(np.asarray(end[:3])-start[:3])*index/steps).tolist()
            self.move_pose(f'{label}_{index:02d}',arm,part,pose,spec,payload,contacts,True)

    def pick(self,id,arm,part,spec=None,dependencies=()):
        spec=spec or grasp(part,arm);pose=self.placed[part]
        approach=list(pose);approach[2]+=.13
        # At the open buffer the side grasp can approach from above. A low
        # horizontal pregrasp brings B's elbow into the table; insertion at
        # the bridge uses a separate horizontal, segmented approach below.
        # The material/state barrier guards the entire approach. Waiting only
        # at grasp would let the TPG approach a part before its deposit exists.
        self.move_pose(id+'_approach',arm,part,approach,spec,dependencies=dependencies)
        self.move_pose(id+'_contact',arm,part,pose,spec,contacts=self.tool_contacts(arm,part)+self.resting_contacts(part),slow=True)
        payload=binding(part,*spec)
        self.skill(id,arm,'grasp',part,dependencies=dependencies,payload_after=payload,
                   effects={'payloads':{arm:payload}})
        self.payloads[arm]=payload
        lift=list(pose);lift[2]+=.15
        self.move_pose(id+'_lift',arm,part,lift,spec,payload,self.resting_contacts(part))
        self.clear_contact_hold(id+'_lift',arm,'pickup')
        return id

    def place(self,id,arm,part,goal,spec,action='install',dependencies=(),stage_side=False):
        payload=self.payloads[arm];approach=list(goal)
        if stage_side:approach[1]-=.20;approach[2]+=.001
        else:approach[2]+=.13
        if part=='Gate':
            # Choose the approach from the seated goal's IK branch. The
            # nearest high approach can otherwise choose the opposite elbow
            # branch, which cannot switch inside the narrow wall opening.
            goal_q=self.ik(arm,transform(goal,*spec),payload,self.resting_contacts(part),label=id+'_goal_branch')
            approach_q=self.ik(arm,transform(approach,*spec),payload,seed=goal_q,label=id+'_approach_branch')
            self.path(id+'_approach',arm,approach_q,payload,dependencies=dependencies)
        else:
            self.move_pose(id+'_approach',arm,part,approach,spec,payload,dependencies=dependencies)
        if stage_side:
            # Split the available clearance between the tower and beam, then
            # seat vertically. Avoid dragging the rigid body on the tower.
            raised=list(goal);raised[2]+=.001
            # This is a clearance motion, so no brick/surface contact is
            # permitted until the following vertical seating movement.
            self.slide_pose(id+'_slide',arm,part,approach,raised,spec,payload)
            self.move_pose(id+'_seat',arm,part,goal,spec,payload,self.resting_contacts(part),slow=True)
        elif part=='Gate':
            self.slide_pose(id+'_slide',arm,part,approach,goal,spec,payload,self.resting_contacts(part))
        else:
            self.move_pose(id+'_contact',arm,part,goal,spec,payload,self.resting_contacts(part),slow=True)
        self.skill(id,arm,action,part,goal,dependencies,
                   effects={'placed':{part:goal},'payloads':{arm:None}},
                   force_N=8. if part=='K2' else 0.,load_position=[.24,.20,1.052])
        self.placed[part]=list(goal);self.payloads[arm]=None
        # Retract along the insertion direction while allowing only the released target/tool contact.
        if part=='Gate':
            self.slide_pose(id+'_retreat',arm,part,goal,approach,spec,None,self.tool_contacts(arm,part)+self.resting_contacts(part))
        else:
            self.move_pose(id+'_retreat',arm,part,approach,spec,contacts=self.tool_contacts(arm,part)+self.resting_contacts(part),slow=True)
        self.clear_contact_hold(id+'_retreat',arm)
        return id

    def qualify_transport(self,arm,part,source,goal,spec):
        self.placed[part]=source
        contacts=self.tool_contacts(arm,part)+self.resting_contacts(part)
        q1=self.ik(arm,transform(source,*spec),contacts=contacts,label='qualify_'+part+'_source')
        payload=binding(part,*spec)
        q2=self.ik(arm,transform(goal,*spec),payload,self.resting_contacts(part),q1,label='qualify_'+part+'_goal')
        return {'q_source':q1,'q_goal':q2,'grasp_offset':spec[0],'grasp_quaternion':spec[1],
                'cost':float(np.linalg.norm(np.asarray(q1)-self.home[arm])+np.linalg.norm(np.asarray(q2)-q1))/.6+2.}

    def qualify(self):
        signature=hashlib.sha256((self.sig+inspect.getsource(grasp)+inspect.getsource(Compiler.qualify)+
            inspect.getsource(Compiler.ik)+inspect.getsource(Compiler.resting_contacts)+
            json.dumps(self.cfg['free_q'],sort_keys=True)).encode()).hexdigest()
        previous=self.out/'allocation_problem.json'
        if previous.exists():
            cached=json.loads(previous.read_text(encoding='utf-8'))
            if cached.get('qualification_signature')==signature:
                print('Reusing exact scene/grasp/backend qualification evidence.',flush=True)
                return cached
        supports=[];modes=[];tasks=[];handoffs=[]
        for name,y,yaw,overlap in [('outer_front',-.05,0,.04),('inner_pocket',.26,math.pi,.14),('weak_narrow',-.085,0,.005)]:
            # The outer shoe needs 20mm lateral clearance from A's palm.
            # A zero-clearance nominal tangent was invalid under measured
            # sub-millimetre tracking/compliance error during insertion.
            x=.08 if name=='outer_front' else .06
            goal=[x,y,.772]+R.from_euler('z',yaw).as_quat().tolist()
            evidence=None;failure=None;self.reset()
            if overlap>=.01:
                try:evidence=self.qualify_transport('B','TemporarySupport',self.source['TemporarySupport'],goal,grasp('TemporarySupport','B'))
                except CompileFailure as error:failure=str(error)
            else:failure='contact overlap below10mm'
            for order in ['before_gate','after_gate']:
                supports.append(dict(id=f'{name}:{order}:B',candidate=name,removal_order=order,placer_arm='B',remover_arm='B',
                    feasible=evidence is not None,cost=evidence['cost']*2 if evidence else 0.,rated_load_N=40.,contact_overlap_m=overlap,
                    blocked_after_gate=name=='inner_pocket',goal_pose=goal,endpoint_evidence=evidence,failure=failure))
            self.qual.append({'part':'TemporarySupport','candidate':name,'feasible':evidence is not None,'evidence':evidence,'failure':failure})
            print(json.dumps({'qualified':'support:'+name,'feasible':evidence is not None,'failure':failure}),flush=True)
        # Qualification after the arm is released: use a verified free A home from the design.
        free=self.cfg['free_q']['A']
        for part in ['K1','K2']:
            for kind in ['supply','install']:
                tid=kind+'_'+part;tasks.append({'id':tid,'kind':kind,'part':part})
                for arm in ['A','B']:
                    self.reset();self.q['A']=free;self.support_contacts=[]
                    source=self.source[part] if kind=='supply' else self.buffer[part]
                    goal=self.buffer[part] if kind=='supply' else self.goals[part]
                    if part=='K2':self.placed['K1']=self.goals['K1']
                    spec=grasp(part,arm,kind);evidence=None;failure=None
                    try:evidence=self.qualify_transport(arm,part,source,goal,spec)
                    except CompileFailure as error:failure=str(error)
                    record=dict(id=tid+':'+arm,task=tid,arm=arm,feasible=evidence is not None,
                                cost=evidence['cost'] if evidence else 0.,grasp_offset=spec[0],grasp_quaternion=spec[1],
                                endpoint_evidence=evidence,failure=failure)
                    modes.append(record);self.qual.append(record)
                    print(json.dumps({'qualified':record['id'],'feasible':record['feasible'],'failure':failure}),flush=True)
            handoffs.append(dict(id='handoff_'+part,part=part,supply_task='supply_'+part,install_task='install_'+part,
                                 from_arm='A',to_arm='B',feasible=True,cost=.5))
        tasks.append(dict(id='close_gate',kind='gate',part='Gate'))
        for arm in ['A','B']:
            self.reset();self.q['A']=free;self.support_contacts=[];self.placed['K1']=self.goals['K1'];self.placed['K2']=self.goals['K2']
            spec=grasp('Gate',arm);evidence=None;failure=None
            try:evidence=self.qualify_transport(arm,'Gate',self.source['Gate'],self.goals['Gate'],spec)
            except CompileFailure as error:failure=str(error)
            modes.append(dict(id='close_gate:'+arm,task='close_gate',arm=arm,feasible=evidence is not None,
                              cost=evidence['cost'] if evidence else 0.,grasp_offset=spec[0],grasp_quaternion=spec[1],endpoint_evidence=evidence,failure=failure))
            print(json.dumps({'qualified':'close_gate:'+arm,'feasible':evidence is not None,'failure':failure}),flush=True)
        reaction=(0.45*9.81*.26+(8+.14*9.81)*.46)/.28
        problem={'support_modes':supports,'tasks':tasks,'task_modes':modes,'handoffs':handoffs,'assembly_order':['K1','K2'],
                 'required_support_load_N':reaction*1.5,'minimum_contact_overlap_m':.010,
                 'buffer_capacity':1,'supply_access':{'K1':['A'],'K2':['A']},
                 'required_reaction_N':reaction,'reaction_safety_factor':1.5,'load_balance_weight':.15}
        problem['qualification_signature']=signature
        save(self.out/'candidate_qualification.json',self.qual);save(self.out/'allocation_problem.json',problem)
        return problem

    def compile(self,allocation):
        self.reset();choice=allocation['selected_support'];support_goal=choice['goal_pose'];arm=choice['placer_arm']
        spec=grasp('TemporarySupport',arm)
        self.pick('pick_support',arm,'TemporarySupport',spec)
        # Lower outside the assembly then translate under the bridge with the jack retracted.
        outside=list(support_goal);outside[1]=-.20
        high=list(outside);high[2]+=.10
        payload=self.payloads[arm]
        # Route the bulky fixture around the front-right of the bridge before
        # entering the narrow shared region beside A's supporting forearm.
        route=list(high);route[0]=.40;route[2]=.92
        self.move_pose('support_route_front',arm,'TemporarySupport',route,spec,payload)
        self.move_pose('support_high_approach',arm,'TemporarySupport',high,spec,payload)
        elevated_outside=list(outside);elevated_outside[2]+=.003
        elevated_goal=list(support_goal);elevated_goal[2]+=.003
        self.move_pose('support_lower_outside',arm,'TemporarySupport',elevated_outside,spec,payload,self.resting_contacts('TemporarySupport'),True)
        self.slide_pose('support_slide_in',arm,'TemporarySupport',elevated_outside,elevated_goal,spec,payload,self.resting_contacts('TemporarySupport'))
        self.move_pose('support_set_down',arm,'TemporarySupport',support_goal,spec,payload,self.resting_contacts('TemporarySupport'),True)
        self.skill('place_support',arm,'release_support','TemporarySupport',support_goal,
                   effects={'placed':{'TemporarySupport':support_goal},'payloads':{arm:None}})
        self.placed['TemporarySupport']=support_goal;self.payloads[arm]=None
        self.support_contacts.append(['SupportContactPad','Bridge'])
        self.skill('verify_support',arm,'verify_support','TemporarySupport',
                   effects={'articulations':{'SupportLift':self.extension},
                            'support_contacts':{'fixture':[['SupportContactPad','Bridge']]}},
                   articulation_motion={'SupportLift':{'start':0.,'end':self.extension}})
        self.art['SupportLift']=self.extension
        self.skill('release_A','A','release_arm_support',dependencies=['verify_support'],
                   effects={'support_contacts':{'A':[['A_SupportShoe','Bridge']]}})
        # First lower the load-bearing arm away from the beam; no lateral loaded sliding.
        a_tcp=self.p.fk('A',self.q['A'])['pose'];a_lower=list(a_tcp);a_lower[2]-=.035
        q=self.ik('A',a_lower,contacts=self.support_contacts,label='A_lower_from_support')
        self.path('A_lower_from_support','A',q,contacts=self.support_contacts,slow=True)
        self.tasks[-1]['metadata']['effects']={'support_contacts':{'A':None}}
        self.support_contacts=[['SupportContactPad','Bridge']]
        self.skill('A_deactivate_pad','A','deactivate_arm_pad',effects={'support_contacts':{'A':None}})
        # Withdraw horizontally above the pocket walls before rotating toward
        # the material area. The wrist cannot rotate freely below the bridge.
        a_front=list(a_lower);a_front[1]=-.10
        self.slide_pose('A_exit_from_support','A',None,a_lower,a_front,([0,0,0],[0,0,0,1]),None,self.support_contacts)
        free=self.cfg['free_q']['A']
        self.path('A_free_home','A',free)
        # B leaves its fixture grasp vicinity while retaining physical support.
        self.path('B_clear_fixture','B',self.home['B'],contacts=self.tool_contacts('B','TemporarySupport'))
        self.clear_contact_hold('B_clear_fixture','B')
        mode={c['task']:c for c in allocation['choices']}
        previous_install=None;previous_buffer_pick=None
        for part in ['K1','K2']:
            supplier=allocation['task_assignments']['supply_'+part];installer=allocation['task_assignments']['install_'+part]
            smode=mode['supply_'+part];imode=mode['install_'+part]
            sspec=(smode['grasp_offset'],smode['grasp_quaternion']);ispec=(imode['grasp_offset'],imode['grasp_quaternion'])
            self.pick('supply_'+part,supplier,part,sspec)
            if supplier!=installer or sspec!=ispec:
                buffer_dependencies=[previous_buffer_pick] if previous_buffer_pick else []
                self.place('deposit_'+part,supplier,part,self.buffer[part],sspec,'release_buffer',
                           dependencies=buffer_dependencies)
                self.path('supplier_clear_'+part,supplier,free if supplier=='A' else self.home['B'])
                # A buffer handoff starts only after the supplier has cleared
                # its released-part contact. Otherwise a still-valid tool/
                # part contact exemption could mask an unmodelled dual hold.
                pick_dependencies=['deposit_'+part+'_retreat_contact_clear']+([previous_install] if previous_install else [])
                self.pick('pick_'+part,installer,part,ispec,dependencies=pick_dependencies)
                previous_buffer_pick='pick_'+part
            self.place('install_'+part,installer,part,self.goals[part],ispec,'install',
                       dependencies=[previous_install] if previous_install else [],stage_side=part=='K1')
            previous_install='install_'+part
        keeper=allocation['task_assignments']['install_K2']
        self.skill('verify_permanent',keeper,'verify_permanent','Bridge',dependencies=['install_K2'],
                   effects={'articulations':{'SupportLift':0.}},
                   articulation_motion={'SupportLift':{'start':self.extension,'end':0.}})
        self.art['SupportLift']=0.
        def gate():
            garm=allocation['task_assignments']['close_gate'];g=mode['close_gate'];gspec=(g['grasp_offset'],g['grasp_quaternion'])
            gate_dependencies=['verify_permanent']
            if choice['removal_order']=='before_gate':gate_dependencies.append('return_support')
            self.pick('pick_gate',garm,'Gate',gspec,dependencies=gate_dependencies)
            self.place('close_gate',garm,'Gate',self.goals['Gate'],gspec,'close_gate')
        if choice['removal_order']=='after_gate':gate()
        remover=choice['remover_arm'];rspec=grasp('TemporarySupport',remover)
        unload_dependencies=['verify_permanent']
        if choice['removal_order']=='after_gate':unload_dependencies.append('close_gate')
        self.skill('unload_support',remover,'unload_support','TemporarySupport',dependencies=unload_dependencies,effects={'articulations':{'SupportLift':0.}})
        # Regrasp from the same face with actuator already unloaded.
        self.move_pose('remove_support_approach',remover,'TemporarySupport',support_goal,rspec,
                       contacts=self.tool_contacts(remover,'TemporarySupport')+self.resting_contacts('TemporarySupport'))
        pay=binding('TemporarySupport',*rspec)
        self.skill('regrasp_support',remover,'grasp','TemporarySupport',payload_after=pay,effects={'payloads':{remover:pay}})
        self.payloads[remover]=pay
        self.move_pose('raise_unloaded_support',remover,'TemporarySupport',elevated_goal,rspec,pay,self.resting_contacts('TemporarySupport'),True)
        self.slide_pose('extract_support_motion',remover,'TemporarySupport',elevated_goal,elevated_outside,rspec,pay,self.resting_contacts('TemporarySupport'))
        self.skill('extract_support',remover,'extract_support','TemporarySupport',
                   effects={'support_contacts':{'fixture':None}})
        self.support_contacts=[]
        self.move_pose('lift_extracted_support',remover,'TemporarySupport',high,rspec,pay,self.resting_contacts('TemporarySupport'))
        self.place('return_support',remover,'TemporarySupport',self.source['TemporarySupport'],rspec,'return_support')
        if choice['removal_order']=='before_gate':gate()
        for side in ['A','B']:self.path(side+'_final_home',side,free if side=='A' else self.home['B'])
        return self.tasks

KNOWN_REFERENCE_ROOT=Path('backups/base_rotation_baseline_20261104_01/results/apex')
KNOWN_REFERENCE_FILES=('alloc_and_reference.json','reference_tasks.json','source_signature.json',
                       'pipeline_manifest.json','summary.json','execution_validation.json','independent_final_audit.json')


def _value_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _path_length(path):
    return sum(math.sqrt(sum((a-b)**2 for a,b in zip(q,r))) for q,r in zip(path,path[1:]))


def load_known_complete_reference(cfg):
    """Accept one intact successful reference only for unchanged physical inputs.

    Historical success is eligibility evidence, not a replacement for this
    run's geometry oracle, rollout or actual dynamics verification.
    """
    evidence={'eligible':False,'source':'known_complete_reference','files':{}}
    try:
        records={}
        for name in KNOWN_REFERENCE_FILES:
            path=ROOT/KNOWN_REFERENCE_ROOT/name
            raw=path.read_bytes();evidence['files'][str(KNOWN_REFERENCE_ROOT/name).replace('\\','/')]=hashlib.sha256(raw).hexdigest()
            records[name]=json.loads(raw.decode('utf-8'))
        reference=records['alloc_and_reference.json'];tasks=reference['tasks']
        manifest=records['pipeline_manifest.json'];signature=records['source_signature.json']
        summary=records['summary.json'];audit=records['execution_validation.json'];independent=records['independent_final_audit.json']
        if (manifest.get('status')!='complete' or not manifest.get('execution_performed') or
            not all(value.get('success') is True for value in (summary,audit,independent))):
            raise ValueError('baseline_has_no_successful_complete_execution')
        run=manifest.get('run_id')
        if not run or any(value.get('run_id')!=run for value in (summary,audit,independent)):
            raise ValueError('baseline_run_identity_mismatch')
        if (len(tasks)!=134 or summary.get('reference_task_count')!=134 or
            tasks!=records['reference_tasks.json'] or _value_hash(tasks)!=manifest.get('reference_signature') or
            reference.get('scene_signature')!=manifest.get('scene_signature')):
            raise ValueError('baseline_complete_reference_identity_mismatch')
        saved_files=signature['inputs']['files'];manifest_files=manifest['input_fingerprint']['files']
        physical=['scenes/10_Scene3_Dynamic.ttt',*[name for name in saved_files if name.startswith('config/')]]
        if not {'config/apex_runtime.json','config/dynamic_homes.json','config/experiment_definition.json','config/scene_manifest.json'}<=set(physical):
            raise ValueError('baseline_missing_physical_config_fingerprints')
        current_configs={str(path.relative_to(ROOT)).replace('\\','/') for path in (ROOT/'config').glob('*.json')}
        if current_configs!={name for name in physical if name.startswith('config/')}:
            raise ValueError('physical_config_file_set_changed')
        for name in physical:
            actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if actual!=saved_files[name] or actual!=manifest_files.get(name):
                raise ValueError('physical_input_changed:'+name)
        audited_scene=independent.get('source_scene_sha256') if 'source_scene_sha256' in independent else independent.get('delivery',{}).get('original_physical_scene_sha256')
        if audited_scene!=saved_files['scenes/10_Scene3_Dynamic.ttt']:
            raise ValueError('baseline_audited_scene_mismatch')
        old_cfg=signature['resolved_scene']
        for key in ('initial','goal','sizes','runtime','free_q','initial_support_contacts','dt'):
            if cfg.get(key)!=old_cfg.get(key):raise ValueError('resolved_physical_configuration_changed:'+key)
        q={arm:list(cfg['arms'][arm]['home_q']) for arm in ('A','B')};seen=set()
        for task in tasks:
            ident,arm,path=task['id'],task['arm'],task['q_path']
            if ident in seen or arm not in q or not path:raise ValueError('invalid_reference_task_structure')
            if any(dep not in seen for dep in task['dependencies']):raise ValueError('reference_dependency_not_closed:'+ident)
            if not all(len(v)==6 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in v) for v in path):
                raise ValueError('invalid_reference_joint_geometry:'+ident)
            if max(abs(a-b) for a,b in zip(q[arm],path[0]))>1e-8:raise ValueError('reference_joint_chain_not_closed:'+ident)
            if task['kind']!='transit' and any(max(abs(a-b) for a,b in zip(path[0],v))>1e-8 for v in path):
                raise ValueError('reference_skill_changes_joint_state:'+ident)
            seen.add(ident);q[arm]=list(path[-1])
        evidence.update(eligible=True,reason='unchanged_physics_and_intact_complete_reference',run_id=run,
                        reference_signature=manifest['reference_signature'],task_count=len(tasks),
                        source_scene_sha256=saved_files['scenes/10_Scene3_Dynamic.ttt'],
                        original_joint_length_rad=sum(_path_length(t['q_path']) for t in tasks if t['kind']=='transit'))
        return reference,evidence
    except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
        evidence['reason']=str(error)
        return None,evidence


def known_reference_allocation(problem,unrestricted,baseline):
    """Use a supported known plan only as a proved equal-objective tie break."""
    evidence={'accepted':False,'unrestricted_objective':unrestricted.get('objective')}
    if not unrestricted.get('solved') or not unrestricted.get('optimal'):
        return None,{**evidence,'reason':'unrestricted_optimum_not_established'}
    support=baseline['selected_support'];support_id=support['id']
    ids={mode['id'] for mode in problem['support_modes']}
    if support_id not in ids:return None,{**evidence,'reason':'baseline_support_not_in_current_problem'}
    restricted=solve_allocation(problem,banned_modes=sorted(ids-{support_id}))
    evidence.update(restricted_objective=restricted.get('objective'),support_id=support_id,
                    restriction='support_modes_only',restricted_solved=restricted.get('solved',False))
    if (not restricted.get('solved') or not restricted.get('optimal') or
        not math.isfinite(float(restricted['objective'])) or
        abs(float(restricted['objective'])-float(unrestricted['objective']))>1e-8):
        return None,{**evidence,'reason':'baseline_support_is_not_an_unrestricted_objective_tie'}
    if restricted.get('task_assignments')!=baseline.get('task_assignments'):
        return None,{**evidence,'reason':'baseline_task_assignments_changed'}
    current_support=restricted['selected_support']
    for key in ('id','candidate','removal_order','placer_arm','remover_arm','goal_pose'):
        if current_support.get(key)!=support.get(key):return None,{**evidence,'reason':'support_semantics_changed:'+key}
    for key in ('grasp_offset','grasp_quaternion'):
        if current_support.get('endpoint_evidence',{}).get(key)!=support.get('endpoint_evidence',{}).get(key):
            return None,{**evidence,'reason':'support_grasp_changed:'+key}
    original={mode['task']:mode for mode in baseline['choices']}
    for mode in restricted['choices']:
        prior=original.get(mode['task'])
        if not prior or any(mode.get(key)!=prior.get(key) for key in ('id','task','arm','grasp_offset','grasp_quaternion')):
            return None,{**evidence,'reason':'task_grasp_or_mode_changed:'+mode['task']}
    if set(restricted['selected_mode_ids'])!=set(baseline['selected_mode_ids']):
        return None,{**evidence,'reason':'baseline_selected_mode_set_changed'}
    evidence.update(accepted=True,reason='known_complete_reference_at_equal_unrestricted_objective')
    restricted['reference_tie_break']=evidence
    restricted['optimal_scope']='support-constrained MILP; objective equality independently checked against unrestricted optimum'
    return restricted,evidence


def refine_known_complete_reference(compiler,reference):
    """Keep the entire IK/skill/dependency chain; improve only bent polylines."""
    tasks=deepcopy(reference['tasks']);compiler.reset();records=[]
    q=deepcopy(compiler.home);payloads={'A':None,'B':None};placed=deepcopy(compiler.source);art={'SupportLift':0.}
    for task in tasks:
        arm=task['arm'];other='B' if arm=='A' else 'A';md=task['metadata'];path=task['q_path']
        if max(abs(a-b) for a,b in zip(q[arm],path[0]))>1e-8:
            raise CompileFailure(task['id'],{'error':'Known reference joint chain does not close'})
        if task.get('payload')!=payloads[arm] or md.get('placed')!=placed or md.get('options',{}).get('articulations')!=art:
            raise CompileFailure(task['id'],{'error':'Known reference payload/world/articulation state does not close'})
        if task['kind']=='transit':
            cost=_path_length(path);chord=math.sqrt(sum((a-b)**2 for a,b in zip(path[0],path[-1])))
            old_planner=md.get('planner')
            if cost>chord+1e-7:
                options=deepcopy(md.get('options',{}))
                options.update(other_q=q[other],other_payload=deepcopy(payloads[other]) or {},
                    contact_exclusions=deepcopy(md.get('contact_exclusions',[])),
                    contact_tolerance=md.get('contact_tolerance',.002),joint_bound_margin=1.2,
                    direct_first=False,max_rrt_candidates=3,shortcut_max_trials=96,shortcut_budget_s=.6,
                    candidate_paths=[{'source':'known_complete_reference:'+task['id'],'path':deepcopy(path)}])
                print(json.dumps({'refining_fixed_reference':task['id'],'baseline_cost':cost}),flush=True)
                result=compiler.p.plan(arm,path[0],path[-1],task.get('payload'),md['placed'],solve_time=3.,options=options)
                candidates=result.get('stats',{}).get('reference_candidates',[])
                original_valid=bool(candidates and candidates[0].get('valid') is True)
                if not original_valid:
                    # An unsuccessful optimization must not discard a whole
                    # valid chain. Obtain a fresh dense certificate of the
                    # original before retaining it; never trust old success.
                    dense=[path[0]]
                    for a,b in zip(path,path[1:]):
                        steps=max(1,math.ceil(math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))/.005))
                        dense.extend([[x+(y-x)*i/steps for x,y in zip(a,b)] for i in range(1,steps+1)])
                    checked=0
                    for offset in range(0,len(dense),128):
                        samples=[{arm:v,other:q[other],'payloads':deepcopy(payloads),
                                  'placed':deepcopy(md['placed']),'options':deepcopy(options)}
                                 for v in dense[offset:offset+128]]
                        answers=compiler.p.validate_samples(samples)
                        if len(answers)!=len(samples) or any(answer.get('ok') is not True for answer in answers):
                            raise CompileFailure(task['id'],{'error':'Original complete-reference path failed current-context revalidation',
                                                            'batch_start':offset,'answers':answers,'optimization_result':result})
                        checked+=len(samples)
                    result.setdefault('stats',{})['original_reference_validation']={'valid':True,'samples':checked,'step_rad':.005}
                selected=result.get('path',[]) if result.get('ok') else []
                endpoints_ok=bool(selected and
                    max(abs(a-b) for a,b in zip(selected[0],path[0]))<=1e-8 and
                    max(abs(a-b) for a,b in zip(selected[-1],path[-1]))<=1e-8)
                new_cost=_path_length(selected) if endpoints_ok else math.inf
                if new_cost>cost+1e-8:
                    selected=path;new_cost=cost;algorithm='ValidatedReferencePath';source='revalidated_original_reference'
                else:
                    algorithm=result.get('algorithm','RRTConnect');source=result.get('path_source','rrt_connect')
                task['q_path']=deepcopy(selected);task['duration']=max(.1,new_cost/.4)
                md.update(planner=algorithm,path_source=source,planning_stats={
                    'source':'known_complete_reference','fixed_endpoints':True,
                    'baseline_joint_length_rad':cost,'selected_joint_length_rad':new_cost})
                compiler.calls.append({'type':algorithm,'label':task['id'],'ok':True,'cached':False,'stats':result.get('stats')})
                records.append({'task_id':task['id'],'baseline_cost':cost,'selected_cost':new_cost,
                                'fixed_endpoints':True,'algorithm':algorithm,'path_source':source,
                                'stats':result.get('stats')})
            else:
                md.update(planner='KnownCompleteReference',path_source='known_complete_reference',
                          planning_stats={'original_planner':old_planner,'current_planner_called':False})
        q[arm]=list(task['q_path'][-1])
        effects=md.get('effects',{})
        placed.update(deepcopy(effects.get('placed',{})))
        payloads.update(deepcopy(effects.get('payloads',{})))
        art.update(deepcopy(effects.get('articulations',{})))
    compiler.tasks=tasks;compiler.q=q
    count=sum(t['kind']=='transit' for t in tasks)
    stats={'source':'known_complete_reference','path_count':count,'planner_query_count':len(records),
           'unchanged_reference_path_count':count-len(records),'refinements':records,
           'rrt_path_count':sum(r['algorithm']=='RRTConnect' for r in records),
           'direct_path_count':sum(r['algorithm']=='CollisionCheckedJointLine' for r in records),
           'reference_path_count':sum(r['algorithm']=='ValidatedReferencePath' for r in records),
           'ompl_solve_call_count':sum(len(r['stats'].get('candidates',[])) for r in records),
           'calls':compiler.calls}
    return tasks,stats


def compile_reference(sim,client,cfg,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    with MotionPlanner(sim,client,cfg) as planner:
        compiler=Compiler(sim,planner,cfg,out)
        problem=compiler.qualify();failures=[];banned=[]
        unrestricted=solve_allocation(problem)
        save(out/'allocation_unrestricted.json',unrestricted)
        baseline,known_evidence=load_known_complete_reference(cfg)
        if baseline is not None:
            allocation,tie=known_reference_allocation(problem,unrestricted,baseline['allocation'])
            known_evidence['allocation_tie_break']=tie
            if allocation is not None:
                try:
                    # Preserve the accepted 317-second chain outside the three
                    # requested local windows. Fresh rollout checks all paths;
                    # unrelated random RRT refinements add no evidence here.
                    tasks=deepcopy(baseline['tasks']);compiler.reset()
                    for task in tasks:
                        if task['kind']=='transit':
                            md=task['metadata'];prior_planner=md.get('planner')
                            md.update(planner='KnownCompleteReference',path_source='known_complete_reference',
                                      planning_stats={'prior_planner':prior_planner,'current_planner_called':False})
                    motion_stats={'source':'known_complete_reference','path_count':sum(t['kind']=='transit' for t in tasks),
                                  'planner_query_count':0,'unchanged_reference_path_count':sum(t['kind']=='transit' for t in tasks),
                                  'rrt_path_count':0,'direct_path_count':0,'reference_path_count':0,
                                  'ompl_solve_call_count':0,'calls':[]}
                    from .local_routes import apply_local_routes
                    tasks,local_routes=apply_local_routes(compiler,baseline,tasks)
                    motion_stats['local_route_refinement']=local_routes
                    changed_ids={row['task_id'] for row in local_routes['changed_task_checks']}
                    motion_stats['revalidated_local_path_count']=sum(t['kind']=='transit' and t['id'] in changed_ids for t in tasks)
                    motion_stats['unchanged_reference_path_count']-=motion_stats['revalidated_local_path_count']
                    save(out/'local_route_validation.json',local_routes)
                    known_evidence['selected']=True
                    known_evidence['planning_timing']=cfg.get('planning_timing',{'vmax':.6,'amax':1.2,'jmax':8.})
                    result={'tasks':tasks,'allocation':allocation,'problem':problem,'motion_stats':motion_stats,
                            'planning_failures':failures,'planner_info':planner.info,
                            'source':'known_complete_reference_with_revalidated_local_routes','known_reference':known_evidence}
                    save(out/'allocation.json',allocation);save(out/'reference_tasks.json',tasks)
                    save(out/'reference_tasks_partial.json',{'stage':'complete_known_reference','tasks':tasks})
                    save(out/'motion_stats.json',motion_stats);save(out/'known_reference_selection.json',known_evidence)
                    return result
                except CompileFailure as error:
                    failures.append({'stage':error.label,'source':'known_complete_reference','evidence':error.evidence})
                    known_evidence.update(selected=False,reason='known_reference_current_geometry_revalidation_failed')
        save(out/'known_reference_selection.json',known_evidence)
        for attempt in range(12):
            allocation=unrestricted if attempt==0 else solve_allocation(problem,banned_combinations=banned)
            save(out/'allocation.json',allocation)
            if not allocation.get('solved'):raise RuntimeError('No geometry-qualified ILP allocation; see allocation_problem.json')
            print(json.dumps({'allocation_support':allocation['selected_support']['id'],'assignments':allocation['task_assignments']}),flush=True)
            try:tasks=compiler.compile(allocation)
            except CompileFailure as error:
                failures.append({'attempt':attempt,'stage':error.label,'evidence':error.evidence,'selected_modes':allocation['selected_mode_ids']})
                save(out/'planning_failures.json',failures);banned.append(allocation['selected_mode_ids'])
                print(str(error),flush=True);continue
            result={'tasks':tasks,'allocation':allocation,'problem':problem,
                    'motion_stats':{'path_count':sum(c['type'] in ('RRTConnect','CollisionCheckedJointLine','ValidatedReferencePath') for c in compiler.calls),
                                    'rrt_path_count':sum(c['type']=='RRTConnect' for c in compiler.calls),
                                    'direct_path_count':sum(c['type']=='CollisionCheckedJointLine' for c in compiler.calls),
                                    'reference_path_count':sum(c['type']=='ValidatedReferencePath' for c in compiler.calls),
                                    'calls':compiler.calls},
                    'planning_failures':failures,'planner_info':planner.info,
                    'source':'fresh_compilation','known_reference':known_evidence}
            save(out/'reference_tasks.json',tasks);save(out/'motion_stats.json',result['motion_stats']);return result
        raise RuntimeError('All bounded task/motion retries failed; no unplanned path was substituted')
