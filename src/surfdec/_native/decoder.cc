// Evidence-conditioned correlated matching on top of PyMatching's sparse blossom.
//
// The matching core (flooder, matcher, search) is PyMatching v2.4.0, vendored
// unmodified under src/surfdec/_vendor (Apache License 2.0). The functions in
// the "adapted from PyMatching" block below are adapted from
// pymatching/sparse_blossom/driver/mwpm_decoding.cc (Apache License 2.0).
//
// Per shot, edge weights are overridden in place (pointers into the graphs'
// neighbor weight arrays) and restored afterwards, as PyMatching does for its
// correlated decoding.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pymatching/sparse_blossom/flooder/graph.h"
#include "pymatching/sparse_blossom/flooder/graph_flooder.h"
#include "pymatching/sparse_blossom/matcher/mwpm.h"
#include "pymatching/sparse_blossom/search/search_flooder.h"
#include "pymatching/sparse_blossom/search/search_graph.h"

namespace py = pybind11;

namespace {

// As in PyMatching's user_graph.h.
constexpr pm::weight_int NUM_DISTINCT_WEIGHTS = 1 << (sizeof(pm::weight_int) * 8 - 8);

// ---- adapted from PyMatching (mwpm_decoding.cc) ----------------------------

void process_timeline_until_completion(pm::Mwpm& mwpm, const std::vector<uint64_t>& detection_events) {
    mwpm.flooder.queue.cur_time = 0;
    for (auto& detection : detection_events) {
        mwpm.create_detection_event(&mwpm.flooder.graph.nodes[detection]);
    }
    while (true) {
        auto event = mwpm.flooder.run_until_next_mwpm_notification();
        if (event.event_type == pm::NO_EVENT)
            break;
        mwpm.process_event(event);
    }
    if (mwpm.node_arena.allocated.size() != mwpm.node_arena.available.size()) {
        mwpm.reset();
        throw std::invalid_argument("No perfect matching could be found.");
    }
}

pm::obs_int decode_obs(pm::Mwpm& mwpm, const std::vector<uint64_t>& detection_events) {
    process_timeline_until_completion(mwpm, detection_events);
    pm::MatchingResult res;
    for (auto& i : detection_events) {
        if (mwpm.flooder.graph.nodes[i].region_that_arrived)
            res += mwpm.shatter_blossom_and_extract_matches(mwpm.flooder.graph.nodes[i].region_that_arrived_top);
    }
    return res.obs_mask;
}

inline void flip_edge(const pm::SearchGraphEdge& edge) {
    edge.detector_node->neighbor_markers[edge.neighbor_index] ^= pm::FLIPPED;
    auto neighbor = edge.detector_node->neighbors[edge.neighbor_index];
    if (neighbor) {
        auto idx_from_neighbor = neighbor->index_of_neighbor(edge.detector_node);
        neighbor->neighbor_markers[idx_from_neighbor] ^= pm::FLIPPED;
    }
}

// Decodes to the set of edges (by edge id) on the shortest paths between
// matched detection events; edges covered an even number of times cancel.
void decode_to_edges(
    pm::Mwpm& mwpm,
    const std::vector<uint64_t>& detection_events,
    const std::vector<std::vector<int32_t>>& search_neighbor_edge,
    std::vector<pm::SearchGraphEdge>& scratch,
    std::vector<int32_t>& out) {
    process_timeline_until_completion(mwpm, detection_events);
    mwpm.flooder.match_edges.clear();
    for (auto& i : detection_events) {
        if (mwpm.flooder.graph.nodes[i].region_that_arrived)
            mwpm.shatter_blossom_and_extract_match_edges(
                mwpm.flooder.graph.nodes[i].region_that_arrived_top, mwpm.flooder.match_edges);
    }
    scratch.clear();
    auto* base = &mwpm.flooder.graph.nodes[0];
    for (const auto& match_edge : mwpm.flooder.match_edges) {
        size_t node_from = match_edge.loc_from - base;
        size_t node_to = match_edge.loc_to ? match_edge.loc_to - base : SIZE_MAX;
        mwpm.search_flooder.iter_edges_on_shortest_path_from_middle(node_from, node_to, [&](const pm::SearchGraphEdge& e) {
            flip_edge(e);
            scratch.push_back(e);
        });
    }
    auto* sbase = &mwpm.search_flooder.graph.nodes[0];
    out.clear();
    for (auto& e : scratch) {
        if (e.detector_node->neighbor_markers[e.neighbor_index] & pm::FLIPPED) {
            flip_edge(e);
            out.push_back(search_neighbor_edge[e.detector_node - sbase][e.neighbor_index]);
        }
    }
}

// ---- end of adapted code ----------------------------------------------------

struct HalfEdges {
    pm::weight_int* f0 = nullptr;
    pm::weight_int* f1 = nullptr;
    pm::weight_int* s0 = nullptr;
    pm::weight_int* s1 = nullptr;
};

struct Graphs {
    pm::Mwpm mwpm;
    std::vector<HalfEdges> half;                            // by edge id
    std::vector<std::vector<int32_t>> search_neighbor_edge;  // [node][neighbor index] -> edge id
    std::vector<std::pair<pm::weight_int*, pm::weight_int>> previous;

    void set(int32_t e, pm::weight_int w, bool search) {
        auto& h = half[e];
        previous.emplace_back(h.f0, *h.f0);
        *h.f0 = w;
        if (h.f1) {
            previous.emplace_back(h.f1, *h.f1);
            *h.f1 = w;
        }
        if (search) {
            previous.emplace_back(h.s0, *h.s0);
            *h.s0 = w;
            if (h.s1) {
                previous.emplace_back(h.s1, *h.s1);
                *h.s1 = w;
            }
        }
    }
    void undo() {
        for (auto it = previous.rbegin(); it != previous.rend(); ++it)
            *it->first = it->second;
        previous.clear();
    }
};

enum Mode : int {
    MATCH = 0,          // one pass, full weights (sanity check against PyMatching)
    CORRELATED = 1,     // PyMatching-style two-pass correlated matching
    EVIDENCE_1 = 2,     // one pass, evidence-conditioned weights
    EVIDENCE_CORR = 3,  // pass 1 full weights; pass 2 evidence-conditioned + correlations
    EVIDENCE_BOTH = 4,  // both passes evidence-conditioned; pass 2 adds correlations
    SINGLE_CORR = 5,    // ablation: both passes on the single-component graph, no evidence;
                        // pass 2 adds correlations (first pass without hyperedges, as in Harmony)
};

class Decoder {
   public:
    Decoder(
        size_t num_detectors,
        size_t num_observables,
        py::array_t<int64_t> edge_u,
        py::array_t<int64_t> edge_v,
        py::array_t<uint64_t> edge_obs,
        py::array_t<double> q_full,
        py::array_t<double> q_single,
        py::array_t<int64_t> mech_ptr,
        py::array_t<int64_t> mech_edges,
        py::array_t<double> mech_p,
        int mode,
        double cap,
        double p_floor,
        bool partners_only)
        : num_detectors(num_detectors), mode(mode), partners_only(partners_only) {
        auto eu = edge_u.unchecked<1>();
        auto ev = edge_v.unchecked<1>();
        auto eo = edge_obs.unchecked<1>();
        auto qf = q_full.unchecked<1>();
        auto qs = q_single.unchecked<1>();
        auto mp = mech_ptr.unchecked<1>();
        auto me = mech_edges.unchecked<1>();
        auto mpr = mech_p.unchecked<1>();
        num_edges = eu.shape(0);
        size_t num_mechs = mpr.shape(0);

        u.resize(num_edges);
        v.resize(num_edges);
        obs.resize(num_edges);
        q_single_.resize(num_edges);
        for (size_t e = 0; e < num_edges; e++) {
            u[e] = eu(e);
            v[e] = ev(e);
            obs[e] = eo(e);
            q_single_[e] = std::max(qs(e), p_floor);
        }
        mech_begin.resize(num_mechs + 1);
        for (size_t h = 0; h <= num_mechs; h++)
            mech_begin[h] = mp(h);
        mech_comp.resize(me.shape(0));
        for (py::ssize_t i = 0; i < me.shape(0); i++)
            mech_comp[i] = me(i);
        mech_prob.resize(num_mechs);
        for (size_t h = 0; h < num_mechs; h++)
            mech_prob[h] = mpr(h);

        // Edge -> multi-component mechanisms containing it; node -> incident edges.
        edge_mechs.resize(num_edges);
        for (size_t h = 0; h < num_mechs; h++)
            for (int64_t i = mech_begin[h]; i < mech_begin[h + 1]; i++)
                edge_mechs[mech_comp[i]].push_back((int32_t)h);
        node_edges.resize(num_detectors);
        for (size_t e = 0; e < num_edges; e++) {
            node_edges[u[e]].push_back((int32_t)e);
            if (v[e] >= 0)
                node_edges[v[e]].push_back((int32_t)e);
        }

        // Implied probabilities (PyMatching's rule): conditioned on e1 being in
        // the error, e2 gets min(cap, P(e1 and e2) / P(e1)).
        std::vector<double> wf(num_edges), ws(num_edges);
        double max_w = 0;
        for (size_t e = 0; e < num_edges; e++) {
            wf[e] = to_w(std::max(qf(e), p_floor));
            ws[e] = to_w(q_single_[e]);
            max_w = std::max(max_w, std::max(wf[e], ws[e]));
        }
        implied.resize(num_edges);
        std::vector<double> joint(num_edges, 0.0);
        std::vector<int32_t> seen;
        for (size_t e1 = 0; e1 < num_edges; e1++) {
            seen.clear();
            for (int32_t h : edge_mechs[e1]) {
                for (int64_t i = mech_begin[h]; i < mech_begin[h + 1]; i++) {
                    int32_t e2 = mech_comp[i];
                    if ((size_t)e2 == e1)
                        continue;
                    if (joint[e2] == 0.0)
                        seen.push_back(e2);
                    joint[e2] = xor_p(joint[e2], mech_prob[h]);
                }
            }
            for (int32_t e2 : seen) {
                double q = std::min(cap, joint[e2] / std::max(qf(e1), p_floor));
                joint[e2] = 0.0;
                if (q > 0)
                    implied[e1].push_back({e2, to_w(q)});
            }
        }
        norm = (double)(NUM_DISTINCT_WEIGHTS - 1) / max_w;

        full = build(wf);
        if (mode != MATCH && mode != CORRELATED)
            single = build(ws);

        implied_int.resize(num_edges);
        for (size_t e = 0; e < num_edges; e++)
            for (auto& [e2, w] : implied[e])
                implied_int[e].push_back({e2, to_int(w)});

        flipped.assign(num_detectors, 0);
        mech_stamp.assign(num_mechs, 0);
        edge_stamp.assign(num_edges, 0);
        edge_q.assign(num_edges, 0.0);
        edge_w.assign(num_edges, 0);
    }

    py::array_t<uint8_t> decode_batch(py::array_t<uint8_t> bit_packed_dets, size_t num_obs_bytes) {
        auto in = bit_packed_dets.unchecked<2>();
        size_t shots = in.shape(0);
        size_t det_bytes = in.shape(1);
        py::array_t<uint8_t> out({shots, num_obs_bytes});
        auto o = out.mutable_unchecked<2>();
        std::vector<uint64_t> dets;
        for (size_t s = 0; s < shots; s++) {
            dets.clear();
            for (size_t b = 0; b < det_bytes; b++) {
                uint8_t byte = in(s, b);
                while (byte) {
                    int bit = __builtin_ctz(byte);
                    size_t d = b * 8 + bit;
                    if (d < num_detectors)
                        dets.push_back(d);
                    byte &= byte - 1;
                }
            }
            pm::obs_int mask = decode(dets);
            for (size_t k = 0; k < num_obs_bytes; k++)
                o(s, k) = (uint8_t)(mask >> (8 * k));
        }
        return out;
    }

    pm::obs_int decode(const std::vector<uint64_t>& dets) {
        if (dets.empty())
            return 0;
        switch (mode) {
            case MATCH:
                return decode_obs(full->mwpm, dets);
            case CORRELATED: {
                decode_to_edges(full->mwpm, dets, full->search_neighbor_edge, scratch, path);
                apply_implied(*full, false);
                auto r = decode_obs(full->mwpm, dets);
                full->undo();
                return r;
            }
            case EVIDENCE_1: {
                apply_evidence(*single, dets, false);
                auto r = decode_obs(single->mwpm, dets);
                single->undo();
                return r;
            }
            case EVIDENCE_CORR: {
                decode_to_edges(full->mwpm, dets, full->search_neighbor_edge, scratch, path);
                apply_evidence(*single, dets, false);
                apply_implied(*single, false);
                auto r = decode_obs(single->mwpm, dets);
                single->undo();
                return r;
            }
            case SINGLE_CORR: {
                decode_to_edges(single->mwpm, dets, single->search_neighbor_edge, scratch, path);
                apply_implied(*single, false);
                auto r = decode_obs(single->mwpm, dets);
                single->undo();
                return r;
            }
            case EVIDENCE_BOTH: {
                apply_evidence(*single, dets, true);
                decode_to_edges(single->mwpm, dets, single->search_neighbor_edge, scratch, path);
                apply_implied(*single, false);
                auto r = decode_obs(single->mwpm, dets);
                single->undo();
                return r;
            }
        }
        throw std::invalid_argument("unknown mode");
    }

    size_t num_edges_() const {
        return num_edges;
    }

   private:
    static double xor_p(double a, double b) {
        return a * (1 - b) + b * (1 - a);
    }
    static double to_w(double q) {
        q = std::min(q, 0.5);
        return std::log((1 - q) / q);
    }
    pm::weight_int to_int(double w) const {
        return (pm::weight_int)(2 * std::llround(w * norm));
    }

    std::unique_ptr<Graphs> build(const std::vector<double>& w) {
        pm::MatchingGraph mg(num_detectors, 64);
        pm::SearchGraph sg(num_detectors);
        for (size_t e = 0; e < num_edges; e++) {
            auto wi = (pm::signed_weight_int)std::llround(w[e] * norm) * 2;
            std::vector<size_t> obs_idx;
            for (size_t k = 0; k < 64; k++)
                if ((obs[e] >> k) & 1)
                    obs_idx.push_back(k);
            if (v[e] >= 0) {
                mg.add_edge(u[e], v[e], wi, obs_idx);
                sg.add_edge(u[e], v[e], wi, obs_idx);
            } else {
                mg.add_boundary_edge(u[e], wi, obs_idx);
                sg.add_boundary_edge(u[e], wi, obs_idx);
            }
        }
        mg.normalising_constant = norm * 2;
        auto g = std::make_unique<Graphs>();
        g->mwpm = pm::Mwpm(pm::GraphFlooder(std::move(mg)), pm::SearchFlooder(std::move(sg)));
        g->mwpm.flooder.sync_negative_weight_observables_and_detection_events();
        auto& fn = g->mwpm.flooder.graph.nodes;
        auto& sn = g->mwpm.search_flooder.graph.nodes;
        g->half.resize(num_edges);
        g->search_neighbor_edge.resize(num_detectors);
        for (size_t n = 0; n < num_detectors; n++)
            g->search_neighbor_edge[n].assign(sn[n].neighbors.size(), -1);
        for (size_t e = 0; e < num_edges; e++) {
            auto& h = g->half[e];
            if (v[e] >= 0) {
                size_t i = fn[u[e]].index_of_neighbor(&fn[v[e]]);
                size_t j = fn[v[e]].index_of_neighbor(&fn[u[e]]);
                h.f0 = &fn[u[e]].neighbor_weights[i];
                h.f1 = &fn[v[e]].neighbor_weights[j];
                size_t a = sn[u[e]].index_of_neighbor(&sn[v[e]]);
                size_t b = sn[v[e]].index_of_neighbor(&sn[u[e]]);
                h.s0 = &sn[u[e]].neighbor_weights[a];
                h.s1 = &sn[v[e]].neighbor_weights[b];
                g->search_neighbor_edge[u[e]][a] = (int32_t)e;
                g->search_neighbor_edge[v[e]][b] = (int32_t)e;
            } else {
                size_t i = fn[u[e]].index_of_neighbor(nullptr);
                h.f0 = &fn[u[e]].neighbor_weights[i];
                size_t a = sn[u[e]].index_of_neighbor(nullptr);
                h.s0 = &sn[u[e]].neighbor_weights[a];
                g->search_neighbor_edge[u[e]][a] = (int32_t)e;
            }
        }
        return g;
    }

    // Evidence: the base graph `single` carries only single-component
    // mechanisms. A multi-component mechanism whose every component touches a
    // detection event gets its probability back on its component edges.
    void apply_evidence(Graphs& g, const std::vector<uint64_t>& dets, bool search) {
        stamp++;
        touched.clear();
        for (auto d : dets)
            flipped[d] = 1;
        for (auto d : dets) {
            for (int32_t e : node_edges[d]) {
                for (int32_t h : edge_mechs[e]) {
                    if (mech_stamp[h] == stamp)
                        continue;
                    mech_stamp[h] = stamp;
                    // A mechanism counts for component c when every other
                    // component touches a detection event (partners_only), or
                    // when all of its components do.
                    int64_t k = mech_begin[h + 1] - mech_begin[h];
                    int64_t num_touched = 0;
                    int32_t untouched = -1;
                    for (int64_t i = mech_begin[h]; i < mech_begin[h + 1]; i++) {
                        int32_t c = mech_comp[i];
                        if (flipped[u[c]] || (v[c] >= 0 && flipped[v[c]]))
                            num_touched++;
                        else
                            untouched = c;
                    }
                    bool all = num_touched == k;
                    if (!all && !(partners_only && num_touched == k - 1))
                        continue;
                    for (int64_t i = mech_begin[h]; i < mech_begin[h + 1]; i++) {
                        int32_t c = mech_comp[i];
                        if (!all && c != untouched)
                            continue;
                        if (edge_stamp[c] != stamp) {
                            edge_stamp[c] = stamp;
                            edge_q[c] = q_single_[c];
                            touched.push_back(c);
                        }
                        edge_q[c] = xor_p(edge_q[c], mech_prob[h]);
                    }
                }
            }
        }
        for (auto d : dets)
            flipped[d] = 0;
        for (int32_t e : touched) {
            pm::weight_int w = to_int(to_w(edge_q[e]));
            if (w != *g.half[e].f0)
                g.set(e, w, search);
        }
    }

    // Correlations (PyMatching's rule): edges sharing a mechanism with an edge
    // of `path` get the implied weight when it is lower than the current one.
    void apply_implied(Graphs& g, bool search) {
        stamp++;
        touched.clear();
        for (int32_t e1 : path) {
            for (auto& [e2, w] : implied_int[e1]) {
                if (edge_stamp[e2] != stamp) {
                    edge_stamp[e2] = stamp;
                    edge_w[e2] = *g.half[e2].f0;
                    touched.push_back(e2);
                }
                if (w < edge_w[e2])
                    edge_w[e2] = w;
            }
        }
        for (int32_t e : touched)
            if (edge_w[e] != *g.half[e].f0)
                g.set(e, edge_w[e], search);
    }

    size_t num_detectors;
    size_t num_edges;
    int mode;
    bool partners_only;
    double norm;
    std::vector<int64_t> u, v;
    std::vector<uint64_t> obs;
    std::vector<double> q_single_;
    std::vector<int64_t> mech_begin;
    std::vector<int32_t> mech_comp;
    std::vector<double> mech_prob;
    std::vector<std::vector<int32_t>> edge_mechs, node_edges;
    std::vector<std::vector<std::pair<int32_t, double>>> implied;
    std::vector<std::vector<std::pair<int32_t, pm::weight_int>>> implied_int;
    std::unique_ptr<Graphs> full, single;

    // per-shot scratch
    uint64_t stamp = 0;
    std::vector<uint8_t> flipped;
    std::vector<uint64_t> mech_stamp, edge_stamp;
    std::vector<double> edge_q;
    std::vector<pm::weight_int> edge_w;
    std::vector<int32_t> touched, path;
    std::vector<pm::SearchGraphEdge> scratch;
};

}  // namespace

PYBIND11_MODULE(_native, m) {
    py::class_<Decoder>(m, "Decoder")
        .def(
            py::init<size_t, size_t, py::array_t<int64_t>, py::array_t<int64_t>, py::array_t<uint64_t>,
                     py::array_t<double>, py::array_t<double>, py::array_t<int64_t>, py::array_t<int64_t>,
                     py::array_t<double>, int, double, double, bool>(),
            py::arg("num_detectors"), py::arg("num_observables"), py::arg("edge_u"), py::arg("edge_v"),
            py::arg("edge_obs"), py::arg("q_full"), py::arg("q_single"), py::arg("mech_ptr"),
            py::arg("mech_edges"), py::arg("mech_p"), py::arg("mode"), py::arg("cap") = 0.5,
            py::arg("p_floor") = 1e-9, py::arg("partners_only") = false)
        .def("decode_batch", &Decoder::decode_batch, py::arg("bit_packed_dets"), py::arg("num_obs_bytes"))
        .def_property_readonly("num_edges", &Decoder::num_edges_);
}
