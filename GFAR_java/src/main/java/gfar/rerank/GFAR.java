package gfar.rerank;

public class GFAR<G,I,U> extends AlphaGFAR<G,I,U> {
    public GFAR(GroupAspectModel<G, U, I> aspectModel, double lambda, int cutoff, boolean norm) {
        super(aspectModel, 1.0, lambda, cutoff, norm);
    }
}
