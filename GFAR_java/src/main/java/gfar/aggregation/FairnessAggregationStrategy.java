package gfar.aggregation;

import es.uam.eps.ir.ranksys.core.Recommendation;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.*;
import java.util.stream.Collectors;

/**
 * FAIRNESS Aggregation Strategy (FAI) in the paper from:
 * Felfernig et al. Group recommender systems: An introduction. Springer. 2018. Based on the ideas of:
 * Judith Masthoff. 2004. Group modeling: Selecting a sequence of television items to suit a group of viewers.
 * <p>
 * The idea is group members pick their favourite items by turns.
 *
 * @param <U>
 * @param <I>
 * @param <G>
 */
public class FairnessAggregationStrategy<U, I, G> extends AbstractAggregationStrategy<U, I, G> {
    private G groupID;
    private List<U> group_members;
    private long top_N;
    private long seed = 96548118; // Give random seed such that Collections.shuffle uses seed, for reproducibility!

    public FairnessAggregationStrategy(G groupID, List<U> group_members, int top_N) {
        this.groupID = groupID;
        this.group_members = group_members;
        this.top_N = top_N;
    }

    @Override
    public Recommendation<G, I> aggregate(Map<U, Recommendation<U, I>> recommendation) {
        // Shuffle the group members list for random ordering
        Collections.shuffle(group_members, new Random(seed));

        List<I> recommendedItems = new ArrayList<>();
        Object2IntOpenHashMap<I> groupRecs = new Object2IntOpenHashMap<>();
        boolean reverse_order = false;
        int rank = 1;
        while (recommendedItems.size() < top_N) {
            if (reverse_order) {
                for (int i = group_members.size() - 1; i >= 0; i--) {
                    U user = group_members.get(i);
                    if (!recommendation.containsKey(user)) continue;
                    Recommendation<U, I> rec = recommendation.get(user);
                    // Below instead of loading all candidate items we only look at the top_N of each member, we guarantee to fill group's top-N
                    List<Tuple2od<I>> userRec = rec.getItems().stream().limit(top_N).collect(Collectors.toList());
                    for (Tuple2od<I> item : userRec) {
                        if (!groupRecs.containsKey(item.v1)) {
                            recommendedItems.add(item.v1);
                            groupRecs.put(item.v1, rank);
                            rank++;
                            break;
                        }
                    }
                }
            } else {
                for (int i = 0; i < group_members.size(); i++) {
                    U user = group_members.get(i);
                    if (!recommendation.containsKey(user)) continue;
                    Recommendation<U, I> rec = recommendation.get(user);
                    List<Tuple2od<I>> userRec = rec.getItems();
                    for (Tuple2od<I> item : userRec) {
                        if (!groupRecs.containsKey(item.v1)) {
                            recommendedItems.add(item.v1);
                            groupRecs.put(item.v1, rank);
                            rank++;
                            break;
                        }
                    }
                }
            }
            reverse_order = !reverse_order;
            if (recommendedItems.size() == 0) break;
        }


        List<Tuple2od<I>> aggregated_scores = new ArrayList<>();
        recommendedItems.forEach(i -> {
            aggregated_scores.add(new Tuple2od<I>(i, 1));
        });

        return new Recommendation<G, I>(groupID, aggregated_scores.stream().limit(top_N).collect(Collectors.toList()));
    }
}
