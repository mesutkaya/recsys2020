package gfar.metrics;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.AbstractRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.basic.NDCG;
import es.uam.eps.ir.ranksys.metrics.rank.RankingDiscountModel;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * NDCG metric in the paper. NDCG (min or minmax).
 *
 * @param <G>
 * @param <I>
 * @param <U>
 */
public class NDCGGroupFairness<G, I, U> extends AbstractRecommendationMetric<G, I> {

    private final NDCG.NDCGRelevanceModel<U, I> relModel;
    private final int cutoff;
    private final RankingDiscountModel disc;
    private final Map<G, List<U>> groups;
    private final String TYPE;

    public NDCGGroupFairness(NDCG.NDCGRelevanceModel<U, I> relModel, int cutoff, RankingDiscountModel disc, Map<G, List<U>> groups, String type) {
        this.relModel = relModel;
        this.cutoff = cutoff;
        this.disc = disc;
        this.groups = groups;
        TYPE = type;
    }

    @Override
    public double evaluate(Recommendation<G, I> recommendation) {
        List<U> groupMembers = groups.get(recommendation.getUser());
        List<Double> group_val = new ArrayList<>();
        for (U user : groupMembers) {
            NDCG.NDCGRelevanceModel<U, I>.UserNDCGRelevanceModel userRelModel = (NDCG.NDCGRelevanceModel<U, I>.UserNDCGRelevanceModel) relModel.getModel(user);

            double ndcg = 0.0;
            int rank = 0;

            for (Tuple2od<I> pair : recommendation.getItems()) {
                ndcg += userRelModel.gain(pair.v1) * disc.disc(rank);

                rank++;
                if (rank >= cutoff) {
                    break;
                }
            }
            if (ndcg > 0) {
                ndcg /= idcg(userRelModel);
            }
            group_val.add(ndcg);
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

    private double idcg(NDCG.NDCGRelevanceModel.UserNDCGRelevanceModel relModel) {
        double[] gains = relModel.getGainValues();
        Arrays.sort(gains);

        double idcg = 0;
        int n = Math.min(cutoff, gains.length);
        int m = gains.length;

        for (int rank = 0; rank < n; rank++) {
            idcg += gains[m - rank - 1] * disc.disc(rank);
        }

        return idcg;
    }

}
