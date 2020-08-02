package gfar.aggregation;

import es.uam.eps.ir.ranksys.core.Recommendation;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/*
XPO algorithm from the paper Dimitris Sacharidis. 2019. Top-N group recommendations with fairness.
In Procs. of the 34th ACM/SIGAPP Symposium on Applied Computing. 1663–1670.
 */

public class XPO<U, I, G> extends AbstractAggregationStrategy<U, I, G> {
    private G groupID;
    private List<U> group_members;
    private int top_N;
    // Random seed for reproducibility
    private Random random = new Random(1234567L);

    public XPO(G groupID, List<U> group_members, int top_N) {
        this.groupID = groupID;
        this.group_members = group_members;
        this.top_N = top_N;
    }

    @Override
    public Recommendation<G, I> aggregate(Map<U, Recommendation<U, I>> recommendation) {
        Map<I, int[]> itemRankings = new HashMap<>();
        Object2IntOpenHashMap<I> NPO = new Object2IntOpenHashMap<>();
        int maxRank = 0;
        for (int i = 0; i < group_members.size(); i++) {
            AtomicInteger rank = new AtomicInteger(0);
            int finalI = i;
            if (!recommendation.containsKey(group_members.get(i))) continue;
            recommendation.get(group_members.get(i)).getItems().stream().limit(top_N).forEach(iTuple2od -> {
                I item = iTuple2od.v1;
                int val = rank.get() + 1;
                if (itemRankings.containsKey(item)) {
                    itemRankings.get(item)[finalI] = val;
                } else {
                    int[] temp = new int[group_members.size()];
                    temp[finalI] = val;
                    itemRankings.put(item, temp);
                }
                rank.getAndIncrement();
            });
            if (rank.get() > maxRank) maxRank = rank.get();
        }

        int x = itemRankings.keySet().size();
        int y = group_members.size();
        int[][] matrix = new int[x][y]; // for each candidate item keeps rankings of each group member!
        Map<Integer, I> indexes = new HashMap<>();
        Map<I, Integer> inverseIndexes = new HashMap<>();
        AtomicInteger index = new AtomicInteger();
        int finalMaxRank = maxRank;
        itemRankings.forEach((item, ints) -> {
            // Here replace 0s in the ranks to maxrank + 1
            for (int i = 0; i < ints.length; i++) {
                if (ints[i] == 0) ints[i] = finalMaxRank + 1;
            }
            indexes.put(index.get(), item);
            inverseIndexes.put(item, index.get());
            matrix[index.get()] = ints;
            index.getAndIncrement();
        });

        for (int i = 0; i < x; i++) {
            for (int j = i + 1; j < x; j++) {
                int[] arr1 = matrix[i];
                int[] arr2 = matrix[j];
                int dominant1 = 0;
                int dominant2 = 0;
                for (int m = 0; m < arr1.length; m++) {
                    int val = arr1[m] - arr2[m];
                    if (val <= 0) dominant1++;
                    if (val >= 0) dominant2++;
                }
                // first item dominants
                if (dominant1 == arr1.length) {
                    NPO.addTo(indexes.get(j), 1);
                    NPO.addTo(indexes.get(i), 0);
                }// second item dominants
                if (dominant2 == arr1.length) {
                    NPO.addTo(indexes.get(i), 1);
                    NPO.addTo(indexes.get(j), 0);
                } else if (dominant1 != arr1.length && dominant2 != arr2.length) {
                    NPO.addTo(indexes.get(i), 0);
                    NPO.addTo(indexes.get(j), 0);
                }
            }
        }
        // Sort XPO based on vals! vals represent number of items that dominates a given item, so 0 means 1-level PO
        List<Tuple2od<I>> candidateItems = new ArrayList<>();
        NPO.forEach((item, integer) -> {
            candidateItems.add(new Tuple2od<I>(item, integer));
        });
        // This keeps the 1-level 2-level 3-level ... N-level PO items
        candidateItems.sort(Comparator.comparingDouble((Tuple2od<I> r) -> r.v2));
        Map<Double, List<I>> NLevelPO = new HashMap<>();
        for (Tuple2od<I> item : candidateItems) {
            double level = item.v2;
            if (NLevelPO.containsKey(level)) {
                NLevelPO.get(level).add(item.v1);
            } else {
                List<I> temp = new ArrayList<>();
                temp.add(item.v1);
                NLevelPO.put(level, temp);
            }
        }

        // Here find X \in [1,N] such that there are at least N items in
        //the x-level Pareto optimal set
        Set<I> finalCandidateItems = new HashSet<>();

        double level = 0.0;
        while (true) {
            if (level == top_N) break;
            if (!NLevelPO.containsKey(level)) {
                level++;
                continue;
            }
            Set<I> temp = new HashSet<>(NLevelPO.get(level));
            finalCandidateItems.addAll(temp);
            if (finalCandidateItems.size() >= top_N) break;
            level++;
        }


        Object2IntOpenHashMap<I> countsInTopN = new Object2IntOpenHashMap<>();
        //initialize random weights to the group members compute the linear combination of the ranks.
        for (int i = 0; i < 10000; i++) { // TODO in the paper it says a large number?
            List<Integer> temp = new ArrayList<>();
            double[] normalized_weights = new double[y];// y is group size

            for (int j = 0; j < y; j++) {
                // TODO maybe change below
                int sample = random.nextInt(10);
                temp.add(sample);
            }
            // Normalize the weights
            int sum = temp.stream().mapToInt(Integer::intValue).sum();
            for (int m = 0; m < temp.size(); m++)
                normalized_weights[m] = temp.get(m) / (double) sum;

            // Now compute linear combination of the ranks of the final candidate items based on weights
            List<Tuple2od<I>> tempCandidateItemVals = new ArrayList<>();
            for (I item : finalCandidateItems) {
                int itemIndex = inverseIndexes.get(item);
                // get ranks vector from the matrix
                int[] arr1 = matrix[itemIndex];
                // dot product of weights and ranks
                int dotProduct = 0;
                for (int j = 0; j < arr1.length; j++)
                    dotProduct += arr1[j] * normalized_weights[j];
                tempCandidateItemVals.add(new Tuple2od<I>(item, dotProduct));
            }
            tempCandidateItemVals.sort(Comparator.comparingDouble((Tuple2od<I> r) -> r.v2));
            tempCandidateItemVals.stream().limit(top_N).forEach(iTuple2od -> {
                countsInTopN.addTo(iTuple2od.v1, 1);
            });
        }

        List<Tuple2od<I>> finalCandidateScores = new ArrayList<>();
        countsInTopN.forEach((item, score) -> {
            finalCandidateScores.add(new Tuple2od<I>(item, score));
        });

        finalCandidateScores.sort(Comparator.comparingDouble((Tuple2od<I> r) -> r.v2).reversed());
        return new Recommendation<G, I>(groupID, finalCandidateScores.stream().limit(top_N).collect(Collectors.toList()));
    }
}
