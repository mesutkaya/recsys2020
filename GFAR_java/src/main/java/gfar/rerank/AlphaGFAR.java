package gfar.rerank;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.LambdaReranker;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.ranksys.core.util.tuples.Tuple2od;

public class AlphaGFAR<G,U,I> extends LambdaReranker<G, I> {
    private final double alpha;
    protected final GroupAspectModel<G,I,U> aspectModel;

    public AlphaGFAR(GroupAspectModel<G,I,U> aspectModel, double alpha, double lambda, int cutoff, boolean norm){
        super(lambda, cutoff, norm);
        this.aspectModel = aspectModel;
        this.alpha = alpha;
    }

    /**
     * Here we use group analogous to user, i.e. we re-rank group recommendations!
     * @param recommendation
     * @param maxLength
     * @return
     */
    @Override
    protected GreedyUserReranker<G, I> getUserReranker(Recommendation<G, I> recommendation, int maxLength) {
        return new GFARAlpha(recommendation, maxLength);
    }

    protected class GFARAlpha extends LambdaUserReranker {
        private final GroupAspectModel<G,I,U>.GroupMembersAspectModel gam;
        private final GroupAspectModel.ItemAspectModel<I,U> iam;
        private final Object2DoubleOpenHashMap<U> redundancy;

        public GFARAlpha(Recommendation<G, I> recommendation, int maxLength){
            super(recommendation, maxLength);
            this.gam = aspectModel.getModel(recommendation.getUser());
            this.iam = gam.getItemAspectModel(recommendation.getItems());
            this.redundancy = new Object2DoubleOpenHashMap<>();
            this.redundancy.defaultReturnValue(1.0);
        }

        @Override
        protected double nov(Tuple2od<I> iv) {
            return gam.getItemIntents(iv.v1)
                    .mapToDouble(u -> {
                        return gam.pu_g(u) * iam.pi_u(iv, u) * redundancy.getDouble(u);
                    })
                    .sum();
        }

        @Override
        protected void update(Tuple2od<I> biv) {
            gam.getItemIntents(biv.v1).sequential()
                    .forEach(u -> {
                        redundancy.put(u, redundancy.getDouble(u) * (1 - alpha * iam.pi_u(biv, u)));
                    });
        }
    }
}
