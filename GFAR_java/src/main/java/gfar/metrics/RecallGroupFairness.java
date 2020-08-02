package gfar.metrics;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.AbstractRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.rel.IdealRelevanceModel;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Recall (min or minmax) metric in the paper
 * @param <G>
 * @param <I>
 * @param <U>
 */
public class RecallGroupFairness<G, I, U> extends AbstractRecommendationMetric<G, I> {
    private final IdealRelevanceModel<U, I> relModel;
    private final int cutoff;
    private final Map<G, List<U>> groups;
    private final String TYPE;

    public RecallGroupFairness(int cutoff, Map<G, List<U>> groups, IdealRelevanceModel<U, I> relModel, String type) {
        this.cutoff = cutoff;
        this.groups = groups;
        this.relModel = relModel;
        this.TYPE = type;
    }

    @Override
    public double evaluate(Recommendation<G, I> recommendation) {
        List<Double> group_val = new ArrayList<>();
        List<U> groupMembers = groups.get(recommendation.getUser());
        for (U user : groupMembers) {
            IdealRelevanceModel.UserIdealRelevanceModel<U, I> userRelModel = relModel.getModel(user);
            int numberOfAllRelevant = relModel.getModel(user).getRelevantItems().size();
            if (numberOfAllRelevant == 0) continue;
            double val = recommendation.getItems().stream()
                    .limit(cutoff)
                    .map(Tuple2od::v1)
                    .filter(userRelModel::isRelevant)
                    .count() / (double) numberOfAllRelevant;//Math.min(cutoff, numberOfAllRelevant)
            group_val.add(val);
        }

        double result = 0.0;
        if (group_val.size() != 0) {
            if (TYPE.equals("MIN"))
                result = group_val.stream().mapToDouble(v -> v).min().getAsDouble();
            else if (TYPE.equals("MIN-MAX")) {
                double max = group_val.stream().mapToDouble(v -> v).max().getAsDouble();
                double min = group_val.stream().mapToDouble(v -> v).min().getAsDouble();
                if (max != 0)
                    result = min / max;
            }
            else if(TYPE.equals("MEAN")){
                result = group_val.stream().mapToDouble(v -> v).average().getAsDouble();
            }
        }
        return result;
    }
}
