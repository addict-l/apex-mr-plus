Normal-mode implementation status: COMPLETE, 2026-10-28.

The complete initial-source-layout -> ILP -> sequential graph -> real RRTConnect -> multimodal TPG -> actual dynamic central execution has passed. The canonical user scene D:/APEX_MR_Plus/APEX_MR_Plus.ttt is identical to scenes/11_Normal_APEX_MR_Planned.ttt and contains INITIAL poses plus the embedded scheduler, rather than a preset final assembly.

Actual audited run: 7 parts, 53 reference actions, 165 TPG nodes, 176 edges; all completed and all predecessor conditions honored. Actual makespan 143.80 s, maximum goal position error 1.60 mm, orientation error 0.001534 rad, support reaction peak 13.84 N. All connection moment peaks below 0.969 N m (limit 1.2); the 8 N insertion stage measured beam connection peak 0.539 N m. Fifteen tests passed. This is one complete run, not a statistical success-rate claim.

Entry points: Open_APEX_MR_Plus.cmd opens the scene; click CoppeliaSim Play. Run_Normal_Mode.cmd reuses compatible plans, revalidates geometry and executes dynamics. Replan_Normal_Mode.cmd recomputes from candidates. Current no-video acceptance run exported real initial, supporting-insertion and completed PNGs plus TPG, timeline and dynamics figures. No MP4 was generated in that run.

Contact fix is in the actual compiler, not only cached tasks: qualify positive shoe/beam contact patch, outside-lower -> translate 3 mm below beam -> vertical engage; reverse on release. Stage paths are real RRT, normal support speed/acceleration limits .025/.12. Height correction is bounded to 2 mm via virtual IK and dynamic joint target commands. Ready barrier requires measured PadFT reaction. Standalone fresh compiler and actual-state contact fixtures passed, as did complete execution.

Model boundaries: native nominal UR5 mass/inertia, Bullet rigid bodies, ideal dynamic GraspFT and SnapFT constraints, finite collision sampling. Not frictional grasp, elastic LEGO snap physics, all of original APEX-MR, arbitrary-delay collision proof, or capability-degradation recovery. Prior preview remains in backups/environment_preview/APEX_MR_Plus.ttt; failures are diagnostic history, not current results.

Primary evidence: results/normal/summary.json, execution_validation.json, execution.json, pipeline_manifest.json, reference_validation.json, nominal_geometry_validation.json, delivery.json. Details and output links are in README.md and normal/README.md.
