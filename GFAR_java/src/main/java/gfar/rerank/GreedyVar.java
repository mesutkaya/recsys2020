package gfar.rerank;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.LambdaReranker;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * GreedyVar fairness approach in "Fairness-Aware Group Recommendation with Pareto-Efficiency" by Xiao et al. 2017.
 *
 * @param <G>
 * @param <U>
 * @param <I>
 */
public class GreedyVar<G, U, I> extends LambdaReranker<G, I> {
    protected Map<G, List<U>> group_members;
    protected int maxLength;
    protected String RELTYPE;
    protected Map<U, Recommendation<U, I>> individualRecommendations;
    protected Map<U, Object2DoubleOpenHashMap<I>> relevances; //Binary or Borda relevance scores for individuals, based on their individual top-K.

    /**
     * Constructor.
     *
     * @param lambda trade-off parameter of the linear combination
     * @param cutoff how many items are re-ranked by the greedy selection.
     * @param norm   decides whether apply a normalization of the values of the
     */
    public GreedyVar(double lambda, int cutoff, int K, boolean norm, Map<G, List<U>> group_members,
                     Map<U, Recommendation<U, I>> individualRecommendations, String RELTYPE) {
        super(lambda, cutoff, norm);
        this.group_members = group_members;
        this.individualRecommendations = individualRecommendations;
        this.RELTYPE = RELTYPE;
        this.maxLength = K;
        relevances = new HashMap<>();

        individualRecommendations.forEach((u, uiRecommendation) -> {
            Object2DoubleOpenHashMap<I> iRelevances = new Object2DoubleOpenHashMap<>();
            AtomicInteger rank = new AtomicInteger(1);
            uiRecommendation.getItems().stream().limit(maxLength).forEach(iTuple2od -> {
                double binRel = 1.0;
                double bordaRel = maxLength - rank.get();
                rank.getAndIncrement();

                if (RELTYPE.equals("BORDA"))
                    iRelevances.addTo(iTuple2od.v1, bordaRel);
                else if (RELTYPE.equals("BINARY"))
                    iRelevances.addTo(iTuple2od.v1, binRel);

            });
            relevances.put(u, iRelevances);
        });
    }

    @Override
    protected GreedyUserReranker<G, I> getUserReranker(Recommendation<G, I> recommendation, int maxLength) {
        return new UserGreedyVar(recommendation, maxLength);
    }

    public class UserGreedyVar extends LambdaUserReranker {
        private final Object2DoubleOpenHashMap<U> utility_within_group; // This is incremental, will be updated in update method.
        private final G group;

        /**
         * Constructor.
         *
         * @param recommendation input recommendation
         * @param maxLength      maximum length of the re-ranked recommendation
         */
        public UserGreedyVar(Recommendation<G, I> recommendation, int maxLength) {
            super(recommendation, maxLength);
            group = recommendation.getUser();
            utility_within_group = new Object2DoubleOpenHashMap<>();

            utility_within_group.defaultReturnValue(0.0);
            group_members.get(group).forEach(u -> {
                utility_within_group.put(u, 0.0);
            });
        }

        @Override
        protected double nov(Tuple2od<I> iv) {
            // Compute gain in fairness and return.
            List<Double> memberUtilities = new ArrayList<>();
            List<Double> memberUtilitiesNew = new ArrayList<>();
            group_members.get(group).forEach(u -> {
                double increment = 0.0;
                boolean flag = false;
                if (individualRecommendations.containsKey(u)) {
                    flag = true;
                    if (individualRecommendations.get(u).getItems().stream().limit(maxLength).filter(item -> item.v1.equals(iv.v1)).collect(Collectors.toList()).size() > 0)
                        increment = relevances.get(u).getDouble(iv.v1);
                }
                memberUtilities.add(utility_within_group.getDouble(u));

                if (RELTYPE.equals("BORDA")) {
                    double denominator = 1.0;
                    if (flag) denominator = relevances.get(u).values().stream().mapToDouble(v -> v).sum();
                    memberUtilitiesNew.add(utility_within_group.getDouble(u) + increment / denominator);
                } else if (RELTYPE.equals("BINARY"))
                    memberUtilitiesNew.add(utility_within_group.getDouble(u) + increment / (double) maxLength);
            });
            return variance(memberUtilities) - variance(memberUtilitiesNew);
        }

        @Override
        protected void update(Tuple2od<I> bestItemValue) {
            // Update individual utility for each group member here, after selecting a new item greedily!
            group_members.get(group).forEach(u -> {
                double increment = 0.0;
                boolean flag = false;
                if (individualRecommendations.containsKey(u)) {
                    flag = true;
                    if (individualRecommendations.get(u).getItems().stream().limit(maxLength).filter(item -> item.v1.equals(bestItemValue.v1)).collect(Collectors.toList()).size() > 0)
                        increment = relevances.get(u).getDouble(bestItemValue.v1);
                }

                if (RELTYPE.equals("BORDA")) {
                    double denominator = 1.0;
                    if (flag) denominator = relevances.get(u).values().stream().mapToDouble(v -> v).sum();
                    utility_within_group.addTo(u, increment / denominator);
                } else if (RELTYPE.equals("BINARY"))
                    utility_within_group.addTo(u, increment / (double) cutoff);
            });
        }

        protected double variance(List<Double> memberUtil) {
            double mean = memberUtil.stream().mapToDouble(v -> v).average().getAsDouble();
            List<Double> values = new ArrayList<>();
            memberUtil.forEach(aDouble -> {
                values.add(Math.pow(aDouble - mean, 2.0));
            });

            return values.stream().mapToDouble(v -> v).average().getAsDouble();
        }
    }
}
