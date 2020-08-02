package gfar.metrics;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.AbstractRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.basic.NDCG;
import es.uam.eps.ir.ranksys.metrics.rank.RankingDiscountModel;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * NDCG metric in the paper. NDCG (mean).
 * @param <G>
 * @param <I>
 * @param <U>
 */
public class NDCGGroup<G, I, U> extends AbstractRecommendationMetric<G, I> {
    private final NDCG.NDCGRelevanceModel<U, I> relModel;
    private final int cutoff;
    private final RankingDiscountModel disc;
    private final Map<G, List<U>> groups;

    public NDCGGroup(NDCG.NDCGRelevanceModel<U, I> relModel, int cutoff, RankingDiscountModel disc, Map<G, List<U>> groups) {
        this.relModel = relModel;
        this.cutoff = cutoff;
        this.disc = disc;
        this.groups = groups;
    }


    @Override
    public double evaluate(Recommendation<G, I> recommendation) {

        double group_val = 0.0;
        List<U> groupMembers = groups.get(recommendation.getUser());
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
            group_val += ndcg;
        }


        return group_val / groupMembers.size();
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
