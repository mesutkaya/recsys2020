package gfar.aggregation;

import es.uam.eps.ir.ranksys.core.Recommendation;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Average aggregation strategy (AVG in the paper). If some items do not exist for some users it assigns score 0 to them
 * Notice that, if we generate scores for individuals based on their unseen items it is possible to recommend to
 * a group that is seen by some of the group members!
 *
 * @param <U>
 * @param <I>
 * @param <G>
 */
public class AverageScoreAggregationStrategy<U, I, G> extends AbstractAggregationStrategy<U, I, G> {
    private G groupID;
    private List<U> group_members;
    private long top_N;

    public AverageScoreAggregationStrategy(G groupID, List<U> group_members, int top_N) {
        this.groupID = groupID;
        this.group_members = group_members;
        this.top_N = top_N;
    }

    @Override
    public Recommendation<G, I> aggregate(Map<U, Recommendation<U, I>> recommendation) {
        Object2DoubleOpenHashMap<I> summation = new Object2DoubleOpenHashMap<>();

        group_members.forEach(member -> {
            if (recommendation.containsKey(member)) {
                recommendation.get(member).getItems().forEach(iTuple2od -> {
                    I item = iTuple2od.v1;
                    double score = iTuple2od.v2;
                    summation.addTo(item, score);
                });
            }
            else{
                System.out.println(groupID);
            }
        });

        List<Tuple2od<I>> aggregated_scores = new ArrayList<>();
        int group_size = group_members.size();
        summation.forEach((i, aDouble) -> {
            aggregated_scores.add(new Tuple2od<I>(i, aDouble / group_size));
        });

        aggregated_scores.sort(Comparator.comparingDouble((Tuple2od<I> r) -> r.v2)
                .reversed());

        return new Recommendation<G, I>(groupID, aggregated_scores.stream().limit(top_N).collect(Collectors.toList()));
    }
}
