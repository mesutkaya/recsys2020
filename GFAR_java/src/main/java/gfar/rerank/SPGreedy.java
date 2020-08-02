package gfar.rerank;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.LambdaReranker;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * SPGreedy algorithm from the paper "Fairness in Package-to-Group Recommendations", it uses m-proportionality!
 * In our paper the results are for m=1 (single proportionality), the code allows to experiment with different
 * values of m!
 *
 *
 * @param <G>
 * @param <U>
 * @param <I>
 */
public class SPGreedy<G, U, I> extends LambdaReranker<G, I> {
    protected Map<G, List<U>> group_members;
    protected int maxLength;
    protected int prop;
    protected int m; // m-proportionality
    protected Map<U, Recommendation<U, I>> individualRecommendations;

    public SPGreedy(double lambda, int cutoff, int maxLength, boolean norm, Map<G, List<U>> group_members,
                    Map<U, Recommendation<U, I>> individualRecommendations, int m, int delta) {
        super(lambda, cutoff, norm);
        this.group_members = group_members;
        this.individualRecommendations = individualRecommendations;
        this.maxLength = maxLength;
        this.m = m;
        this.prop = delta;
    }

    @Override
    protected GreedyUserReranker<G, I> getUserReranker(Recommendation<G, I> recommendation, int maxLength) {
        return new UserSPGreedy(recommendation, maxLength);
    }

    public class UserSPGreedy extends LambdaUserReranker {
        private final Object2DoubleOpenHashMap<U> utility_within_group; // This is incremental, will be updated in update method.
        private final G group;
        private final int groupSize;

        /**
         * Constructor.
         *
         * @param recommendation input recommendation
         * @param maxLength      maximum length of the re-ranked recommendation
         */
        public UserSPGreedy(Recommendation<G, I> recommendation, int maxLength) {
            super(recommendation, maxLength);
            group = recommendation.getUser();
            this.groupSize = group_members.get(group).size();
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
                if (individualRecommendations.containsKey(u)) {
                    if (individualRecommendations.get(u).getItems().stream().limit(prop).filter(item -> item.v1.equals(iv.v1)).collect(Collectors.toList()).size() > 0)
                        increment = 1.0;
                }
                memberUtilities.add(utility_within_group.getDouble(u));
                memberUtilitiesNew.add(utility_within_group.getDouble(u) + increment);

            });
            double val1 = memberUtilitiesNew.stream().mapToDouble(v -> v).filter(x -> x >= m).count();
            double val2 = memberUtilities.stream().mapToDouble(v -> v).filter(x -> x >= m).count();
            return (val1 - val2) / (double) groupSize;
        }

        @Override
        protected void update(Tuple2od<I> bestItemValue) {
            // Update individual utility for each group member here, after selecting a new item greedily!
            group_members.get(group).forEach(u -> {
                double increment = 0.0;

                if (individualRecommendations.containsKey(u)) {

                    if (individualRecommendations.get(u).getItems().stream().limit(prop).filter(item -> item.v1.equals(bestItemValue.v1)).collect(Collectors.toList()).size() > 0)
                        increment = 1.0;
                }
                utility_within_group.addTo(u, increment);
            });
        }
    }
}
