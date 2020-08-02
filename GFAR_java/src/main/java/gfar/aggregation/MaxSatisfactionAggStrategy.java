package gfar.aggregation;

import es.uam.eps.ir.ranksys.core.Recommendation;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.*;
import java.util.stream.Collectors;

public class MaxSatisfactionAggStrategy<U, I, G> extends AbstractAggregationStrategy<U, I, G> {
    private G groupID;
    private List<U> group_members;
    private long top_N;

    public MaxSatisfactionAggStrategy(G groupID, List<U> group_members, int top_N){
        this.groupID = groupID;
        this.group_members = group_members;
        this.top_N = top_N;
    }

    @Override
    public Recommendation<G, I> aggregate(Map<U, Recommendation<U, I>> recommendation) {
        Map<I, List<Double>> scores = new HashMap<>();

        group_members.forEach(member -> {
            if (recommendation.containsKey(member)) {
                recommendation.get(member).getItems().forEach(iTuple2od -> {
                    I item = iTuple2od.v1;
                    double score = iTuple2od.v2;
                    if(scores.containsKey(item))
                        scores.get(item).add(score);
                    else{
                        List<Double> itemScores = new ArrayList<>();
                        itemScores.add(score);
                        scores.put(item,itemScores);
                    }

                });
            }
            else{
                System.out.println(groupID);
            }
        });
        List<Tuple2od<I>> aggregated_scores = new ArrayList<>();

        scores.forEach((i, doubles) -> {
            double aggScore = 0.0;
            if (doubles.size() == group_members.size()){
                aggScore = doubles.stream().mapToDouble(v -> v).max().getAsDouble();
            }
            aggregated_scores.add(new Tuple2od<I>(i, aggScore));
        });

        aggregated_scores.sort(Comparator.comparingDouble((Tuple2od<I> r) -> r.v2)
                .reversed());

        return new Recommendation<G, I>(groupID, aggregated_scores.stream().limit(top_N).collect(Collectors.toList()));
    }
}
