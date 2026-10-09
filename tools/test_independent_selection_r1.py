from copy import deepcopy
import numpy as np
import pytest
from empirical_independent_selection_r1 import analyse, compare, FRESH, FUTURES, intervals, item_folds, ROLES, weights


def fixture():
    parent = {r:[dict(id=f"{r}-{i}", prompt_sha256=f"prompt-{i}", correct=0) for i in range(8)] for r in ROLES}
    candidates = []
    for j in range(3):
        conditions = {}
        for name in ("original_m1", *FUTURES):
            conditions[name] = dict(items={r:[dict(x, correct=int(j == 0)) for x in values] for r,values in parent.items()}, seed=10+(50000000 if name == FRESH[0] else 100000000 if name == FRESH[1] else 0), continuation_available=1)
        candidates.append(dict(candidate_id=str(j), conditions=conditions))
    return dict(parent=dict(items=parent), candidates=candidates, state_id="s", trajectory_id="t", stream="m/code")


def test_independent_gap_can_be_negative():
    result=compare([0,1,0], [1,0,0], [-.1,.2,0])
    assert result['independent_future_minus_immediate'] == pytest.approx(-.3)
    assert result['independent_future_utility'] == pytest.approx(-.1)


def test_ties_use_uniform_mass():
    assert weights([1,1,0]).tolist() == [.5,.5,0]
    assert compare([1,1,0], [0,0,1], [0,1,2])['immediate_future_utility'] == .5
    with pytest.raises(ValueError): weights([float('nan'),0,0])


def test_primary_selection_does_not_use_scoring_seed():
    r=fixture();before=analyse(r)
    for j,c in enumerate(r['candidates']):
        for items in c['conditions'][FRESH[1]]['items'].values():
            for item in items: item['correct']=int(j==2)
    after=analyse(r)
    # Seed-A selector stays candidate zero when seed-B's winner changes.
    assert before['primary']['independent_future_utility'] == 1
    assert after['primary']['independent_future_utility'] == 0


def test_disjoint_folds_are_stable_under_input_order():
    r=fixture();folds=item_folds(r)
    for role,(a,b) in folds.items():
        assert not set(a)&set(b)
        assert set(a)|set(b)==set(range(8))
    altered=deepcopy(r)
    for role in ROLES: altered['parent']['items'][role].reverse()
    changed=item_folds(altered)
    for role in ROLES:
        ids=lambda rec,indices:{rec['parent']['items'][role][i]['id'] for i in indices}
        assert ids(r,folds[role][0])==ids(altered,changed[role][0])


def test_disjoint_item_primary_avoids_shared_item_winner_advantage():
    r=fixture();folds=item_folds(r)
    for role,(a,b) in folds.items():
        for j,c in enumerate(r['candidates']):
            for name in FRESH:
                for i,item in enumerate(c['conditions'][name]['items'][role]):
                    item['correct']=int((j==0 and i in a) or (j==1 and i in b))
            for item in c['conditions']['original_m1']['items'][role]: item['correct']=int(j==2)
    result=analyse(r)
    assert result['primary']['independent_future_minus_immediate']==0


def test_identity_and_seed_errors_are_rejected():
    r=fixture();r['candidates'][0]['conditions'][FRESH[1]]['seed']=r['candidates'][0]['conditions'][FRESH[0]]['seed']
    with pytest.raises(ValueError): analyse(r)
    r=fixture();r['candidates'][0]['conditions'][FRESH[0]]['items'][ROLES[0]][0]['id']='wrong'
    with pytest.raises(RuntimeError): analyse(r)
    r=fixture();r['parent']['items'][ROLES[0]][0]['id']=r['parent']['items'][ROLES[0]][1]['id']
    with pytest.raises(ValueError): item_folds(r)


def test_intervals_count_trajectories_not_pairs_or_folds():
    rows=[dict(trajectory_id=str(i),stream=str(i//2)) for i in range(8)]
    values=[dict(gap=.2) for _ in rows]
    result=intervals(rows,values)['gap']
    assert result['independent_trajectories']==8
    assert np.allclose(result['ci95'],[.2,.2])
    rows[-1]['trajectory_id']=rows[0]['trajectory_id']
    with pytest.raises(ValueError): intervals(rows,values)
