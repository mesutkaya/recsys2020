package gfar.rerank;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.LambdaReranker;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.*;
import java.util.stream.Collectors;

/**
 * EFGreedy algorithm from the paper "Fairness in Package-to-Group Recommendations",
 * <p>
 *
 * @param <G>
 * @param <U>
 * @param <I>
 */
public class EFGreedy<G, U, I> extends LambdaReranker<G, I> {
    protected Map<G, List<U>> group_members;
    protected int maxLength;
    protected int prop;
    protected int m;
    protected Map<U, Recommendation<U, I>> individualRecommendations;

    public EFGreedy(double lambda, int cutoff, int maxLength, boolean norm, Map<G, List<U>> group_members,
                    Map<U, Recommendation<U, I>> individualRecommendations, int m, int delta) {
        super(lambda, cutoff, norm);
        this.group_members = group_members;
        this.individualRecommendations = individualRecommendations;
        this.maxLength = maxLength;
        this.m = m;
        this.prop = delta; // The max value is the group size!
    }

    @Override
    protected GreedyUserReranker<G, I> getUserReranker(Recommendation<G, I> recommendation, int maxLength) {
        return new UserEFGreedy(recommendation, maxLength);
    }

    public class UserEFGreedy extends LambdaUserReranker {
        private final Object2DoubleOpenHashMap<U> utility_within_group; // This is incremental, will be updated in update method.
        private final G group;
        private final int groupSize;
        private final Map<U, Object2DoubleOpenHashMap<I>> userRanks;

        /**
         * Constructor.
         *
         * @param recommendation input recommendation
         * @param maxLength      maximum length of the re-ranked recommendation
         */
        public UserEFGreedy(Recommendation<G, I> recommendation, int maxLength) {
            super(recommendation, maxLength);
            group = recommendation.getUser();
            this.groupSize = group_members.get(group).size();
            utility_within_group = new Object2DoubleOpenHashMap<>();

            utility_within_group.defaultReturnValue(0.0);
            group_members.get(group).forEach(u -> {
                utility_within_group.put(u, 0.0);
            });

            userRanks = new HashMap<>();
            group_members.get(group).forEach(u -> {
                if (individualRecommendations.containsKey(u)) {
                    List<Tuple2od<I>> temp = individualRecommendations.get(u).getItems().stream().limit(maxLength).collect(Collectors.toList());
                    Object2DoubleOpenHashMap<I> uRanks = new Object2DoubleOpenHashMap<>();

                    for (int i = 0; i < temp.size(); i++) {
                        double rank = maxLength - (i + 1);
                        double score = temp.get(i).v2;
                        uRanks.addTo(temp.get(i).v1, score);
                    }
                    userRanks.put(u, uRanks);
                }
            });
        }

        @Override
        protected double nov(Tuple2od<I> iv) {
            // Compute gain in fairness and return.
            List<Double> memberUtilities = new ArrayList<>();
            List<Double> memberUtilitiesNew = new ArrayList<>();

            List<Tuple2od<U>> uRanks = new ArrayList<>();

            group_members.get(group).forEach(u -> {
                double userRank = 0.0;
                if (individualRecommendations.containsKey(u)) {
                    if (userRanks.get(u).containsKey(iv.v1)) {
                        userRank = userRanks.get(u).getDouble(iv.v1);
                    }
                }
                uRanks.add(new Tuple2od<U>(u, userRank));
            });
            uRanks.sort(Comparator.comparingDouble((Tuple2od<U> r) -> r.v2)
                    .reversed());

            for (int i = 0; i < uRanks.size(); i++) {
                U u = uRanks.get(i).v1;
                double increment = 0.0;
                if (i < prop) {
                    increment = 1.0;
                }
                memberUtilities.add(utility_within_group.getDouble(u));
                memberUtilitiesNew.add(utility_within_group.getDouble(u) + increment);
            }

            double val1 = memberUtilitiesNew.stream().mapToDouble(v -> v).filter(x -> x >= m).count();
            double val2 = memberUtilities.stream().mapToDouble(v -> v).filter(x -> x >= m).count();
            return (val1 - val2) / (double) groupSize;
        }

        @Override
        protected void update(Tuple2od<I> iv) {
            // Update individual utility for each group member here, after selecting a new item greedily!

            List<Tuple2od<U>> uRanks = new ArrayList<>();

            group_members.get(group).forEach(u -> {
                double userRank = 0.0;
                if (individualRecommendations.containsKey(u)) {
                    if (userRanks.get(u).containsKey(iv.v1)) {
                        userRank = userRanks.get(u).getDouble(iv.v1);
                    }
                }
                uRanks.add(new Tuple2od<U>(u, userRank));
            });
            uRanks.sort(Comparator.comparingDouble((Tuple2od<U> r) -> r.v2)
                    .reversed());

            for (int i = 0; i < uRanks.size(); i++) {
                U u = uRanks.get(i).v1;
                double increment = 0.0;
                if (i < prop) {
                    increment = 1.0;
                }
                utility_within_group.addTo(u, increment);
            }
        }
    }
}
