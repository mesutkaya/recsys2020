package gfar.metrics;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.AbstractRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.rel.IdealRelevanceModel;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Zero Recall metric in the paper based on paper "Koen Verstrepen and Bart Goethals. 2015. Top-N Recommendation for
 * Shared Accounts.
 * In Proceedings of the 9th ACM Conference on Recommender Systems (RecSys '15). ACM, New York, NY, USA, 59-66.
 * DOI: https://doi.org/10.1145/2792838.2800170"
 *
 * It measures the fraction of group members for whom no relevant item was retrieved in the top-𝑁_G.
 *
 * @param <G>
 * @param <I>
 * @param <U>
 */
public class ZeroRecallGroupFairness<G, I, U> extends AbstractRecommendationMetric<G, I> {
    private final IdealRelevanceModel<U, I> relModel;
    private final int cutoff;
    private final Map<G, List<U>> groups;

    public ZeroRecallGroupFairness(int cutoff, Map<G, List<U>> groups, IdealRelevanceModel<U, I> relModel) {
        this.cutoff = cutoff;
        this.groups = groups;
        this.relModel = relModel;
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
                    .count() / (double) numberOfAllRelevant;
            group_val.add(val);
        }
        if (group_val.size() == 0) return 0;
        else return group_val.stream().mapToDouble(v -> v).filter(v -> v == 0.0).count() / (double) groupMembers.size();

    }
}
