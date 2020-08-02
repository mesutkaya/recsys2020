package gfar.metrics;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.AbstractRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.rank.RankingDiscountModel;
import es.uam.eps.ir.ranksys.metrics.rel.RelevanceModel;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Discounted First Hit metric. It is the DFH(min or minmax) metric used in the paper.
 *
 * @param <G>
 * @param <I>
 * @param <U>
 */
public class DiscountedFirstHitFairness<G, I, U> extends AbstractRecommendationMetric<G, I> {
    private final int cutoff;
    private final RankingDiscountModel disc;
    private final Map<G, List<U>> groups;
    private final RelevanceModel<U, I> relModel;
    private final String TYPE;

    public DiscountedFirstHitFairness(int cutoff, RankingDiscountModel disc, Map<G, List<U>> groups, RelevanceModel<U, I> relModel, String type) {
        this.cutoff = cutoff;
        this.disc = disc;
        this.groups = groups;
        this.relModel = relModel;
        this.TYPE = type;
    }

    @Override
    public double evaluate(Recommendation<G, I> recommendation) {
        List<U> groupMembers = groups.get(recommendation.getUser());
        List<Double> group_val = new ArrayList<>();


        for (U user : groupMembers) {
            int rank = 0;
            RelevanceModel.UserRelevanceModel<U, I> userRelModel = relModel.getModel(user);
            double val = 0.0;
            for (Tuple2od<I> pair : recommendation.getItems()) {
                if (userRelModel.isRelevant(pair.v1)) {
                    // We find a hit get rank and break
                    val = disc.disc(rank);
                    break;
                }
                rank++;
                if (rank >= cutoff) {
                    break;
                }
            }
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
            } else if (TYPE.equals("MEAN")) {
                result = group_val.stream().mapToDouble(v -> v).average().getAsDouble();
            }
        }
        return result;
    }
}
