import pytest
from empirical_review_dispatch_r1 import trim_completed


def test_completed_and_failed_afterany_jobs_leave_only_active_dependencies():
    args=['--parsable','--dependency=afterany:1:2:3','review']
    states={'1':dict(state='COMPLETED'),'2':dict(state='FAILED'),'3':dict(state='RUNNING')}
    assert trim_completed(args,states)==['--parsable','--dependency=afterany:3','review']
    states['3']['state']='COMPLETED'
    assert trim_completed(args,states)==['--parsable','review']
    assert args[1]=='--dependency=afterany:1:2:3'


def test_unknown_dependencies_never_bypassed():
    with pytest.raises(ValueError): trim_completed(['--dependency=afterany:1'],{})
    with pytest.raises(ValueError): trim_completed(['--dependency=afterok:1'],{'1':dict(state='COMPLETED')})
