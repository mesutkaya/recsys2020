package gfar.metrics;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.AbstractRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.rel.IdealRelevanceModel;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.List;
import java.util.Map;

/**
 * Recall (mean)
 * @param <G>
 * @param <I>
 * @param <U>
 */
public class RecallGroup<G, I, U> extends AbstractRecommendationMetric<G, I> {
    private final IdealRelevanceModel<U, I> relModel;
    private final int cutoff;
    private final Map<G, List<U>> groups;

    public RecallGroup(int cutoff, Map<G, List<U>> groups, IdealRelevanceModel<U, I> relModel){
        this.cutoff = cutoff;
        this.groups = groups;
        this.relModel = relModel;
    }

    @Override
    public double evaluate(Recommendation<G, I> recommendation) {
        double group_val = 0.0;
        List<U> groupMembers = groups.get(recommendation.getUser());
        for(U user: groupMembers){
            IdealRelevanceModel.UserIdealRelevanceModel<U, I> userRelModel = relModel.getModel(user);
            int numberOfAllRelevant = relModel.getModel(user).getRelevantItems().size();
            if (numberOfAllRelevant == 0) continue;
            double val = recommendation.getItems().stream()
                    .limit(cutoff)
                    .map(Tuple2od::v1)
                    .filter(userRelModel::isRelevant)
                    .count() / (double) numberOfAllRelevant; // Math.min(cutoff, numberOfAllRelevant)
            group_val += val;
        }
        return group_val / groupMembers.size();
    }
}
