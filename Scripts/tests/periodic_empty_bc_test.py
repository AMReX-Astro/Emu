'''
Regression checks for the periodic_empty particle boundary condition.

Expects plt00000 (init) and plt00001 (after one step) with particle data in the
current working directory. Two modes:

  default / --case mixed
    x faces are periodic_empty; y/z are periodic.
    +x particles that wrap through the high face reenter emptied.
    Interior +x movers keep N across grid splits.
    +y wraps through periodic keep N.

  --case corner
    x and y faces are periodic_empty; diagonal flight.
    A corner exit reenters in both coordinates with N cleared once and
    momentum signs unchanged.

Created for Emu CI coverage of periodic_empty.
'''

import argparse
import sys
import os
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 '../data_reduction'))
import amrex_plot_tools as amrex

parser = argparse.ArgumentParser()
parser.add_argument(
    '--case', choices=('mixed', 'corner'), default='mixed',
    help='Which one-step scenario to validate.')
parser.add_argument(
    '-na', '--no_assert', action='store_true',
    help='Report failures without raising AssertionError.')
args = parser.parse_args()


def myassert(condition, message):
    if not condition:
        print('FAIL:', message)
        if not args.no_assert:
            assert condition, message


def load(plt):
    idata, rdata = amrex.read_particle_data(plt, ptype='neutrinos')
    return idata, rdata


rkey, _ = amrex.get_particle_keys(2)
Ncomps = [
    'N00_Re', 'N01_Re', 'N01_Im', 'N11_Re',
    'N00_Rebar', 'N01_Rebar', 'N01_Imbar', 'N11_Rebar',
]
ix, iy = rkey['pos_x'], rkey['pos_y']
ipupx, ipupy, ipupt = rkey['pupx'], rkey['pupy'], rkey['pupt']

_, p0 = load('plt00000')
_, p1 = load('plt00001')

myassert(len(p0) == len(p1),
         f'particle count changed {len(p0)} -> {len(p1)}')

# Off-diagonals must survive init (perturbation_type unused).
off0 = np.max(np.abs(p0[:, [rkey[c] for c in
                            ('N01_Re', 'N01_Im', 'N01_Rebar', 'N01_Imbar')]]))
myassert(off0 > 1e-30, 'initial off-diagonals are zero; check particle input')

if args.case == 'mixed':
    is_px = (np.abs(p1[:, ipupy]) < 1e-30) & (p1[:, ipupx] > 0)
    is_py = (np.abs(p1[:, ipupx]) < 1e-30) & (p1[:, ipupy] > 0)
    myassert(is_px.sum() + is_py.sum() == len(p1),
             'unexpected particle directions')

    # After one +x cell of travel the lattice looks the same; wrapped +x
    # particles re-enter at x = 0.5.
    px_wrapped = is_px & (np.abs(p1[:, ix] - 0.5) < 1e-10)
    px_unwrapped = is_px & (np.abs(p1[:, ix] - 0.5) >= 1e-10)
    py_wrapped = is_py & (np.abs(p1[:, iy] - 0.5) < 1e-10)

    myassert(px_wrapped.sum() > 0, 'no +x particles wrapped')
    myassert(px_unwrapped.sum() > 0, 'no interior +x particles')
    myassert(py_wrapped.sum() > 0, 'no +y particles wrapped')

    ref_pup = p1[px_unwrapped][0, [ipupx, ipupy, rkey['pupz'], ipupt]]

    for i in np.where(px_wrapped)[0]:
        for c in Ncomps:
            myassert(abs(p1[i, rkey[c]]) <= 1e-30,
                     f'px wrap id loc {i}: {c}={p1[i, rkey[c]]} not zero')
        pup = p1[i, [ipupx, ipupy, rkey['pupz'], ipupt]]
        myassert(np.max(np.abs(pup - ref_pup)) <= 1e-30,
                 f'px wrap momentum changed: {pup}')
        myassert(0.0 <= p1[i, ix] <= 1.0,
                 f'px wrap not near lo face: x={p1[i, ix]}')

    for i in np.where(px_unwrapped)[0]:
        myassert(abs(p1[i, rkey['N00_Re']]) > 1e-30,
                 f'interior +x false zero at x={p1[i, ix]}')
        pup = p1[i, [ipupx, ipupy, rkey['pupz'], ipupt]]
        myassert(np.max(np.abs(pup - ref_pup)) <= 1e-30,
                 'interior +x momentum changed')

    off_u = np.max(np.abs(
        p1[px_unwrapped][:, [rkey[c] for c in
                            ('N01_Re', 'N01_Im', 'N01_Rebar', 'N01_Imbar')]]))
    myassert(off_u > 1e-30, 'interior +x lost all off-diagonals')

    for i in np.where(py_wrapped)[0]:
        myassert(abs(p1[i, rkey['N00_Re']]) > 1e-30,
                 f'plain periodic +y wrap emptied N00 at y={p1[i, iy]}')
        myassert(p1[i, ipupy] > 0, 'plain periodic +y lost momentum')

    print('PASS mixed: periodic_empty wrap+empty, interior keep-N, '
          'plain periodic keep-N')

else:  # corner
    emptied = np.ones(len(p1), dtype=bool)
    for c in Ncomps:
        emptied &= np.abs(p1[:, rkey[c]]) <= 1e-30

    # With diagonal flight of one cell length, starters with x>3 or y>3 exit.
    expect_empty = (p0[:, ix] > 3.0) | (p0[:, iy] > 3.0)
    myassert(emptied.sum() == expect_empty.sum(),
             f'emptied {emptied.sum()} vs expect {expect_empty.sum()}')

    myassert(np.all(p1[:, ipupx] > 0) and np.all(p1[:, ipupy] > 0),
             'corner case momentum sign flipped')

    ll = emptied & (p1[:, ix] < 1.0) & (p1[:, iy] < 1.0)
    myassert(ll.sum() >= 1, 'no multi-axis reentry into lo-lo corner')

    for i in np.where(ll)[0]:
        myassert(abs(p1[i, ipupx] - p0[0, ipupx]) <= 1e-14 and
                 abs(p1[i, ipupy] - p0[0, ipupy]) <= 1e-14,
                 'corner reentry momentum changed')
        for c in Ncomps:
            myassert(abs(p1[i, rkey[c]]) <= 1e-30,
                     f'corner reentry {c} not zero')

    keep = ~emptied
    myassert(keep.sum() > 0, 'all particles emptied unexpectedly')
    myassert(np.max(np.abs(p1[keep][:, rkey['N00_Re']])) > 1e-30,
             'kept particles lost N00')
    off_k = np.max(np.abs(
        p1[keep][:, [rkey[c] for c in
                     ('N01_Re', 'N01_Im', 'N01_Rebar', 'N01_Imbar')]]))
    myassert(off_k > 1e-30, 'kept particles lost off-diagonals')

    print('PASS corner: multi-axis periodic_empty wrap, single N clear, '
          'no momentum flip')
