package gfar.rerank;

import es.uam.eps.ir.ranksys.core.model.UserModel;
import es.uam.eps.ir.ranksys.diversity.intentaware.AspectModel;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

public abstract class GroupAspectModel<G,I,U> extends UserModel<G> {

    protected GroupIntentModel<G,I,U> intentModel;
    public GroupAspectModel(GroupIntentModel<G,I,U> iModel){this.intentModel = iModel;}

    @SuppressWarnings("unchecked")
    @Override
    public GroupMembersAspectModel getModel(G group) {
        return (GroupMembersAspectModel) super.getModel(group);
    }

    /**
     * User aspect model for {@link GroupAspectModel}.
     */
    public abstract class GroupMembersAspectModel implements GroupIntentModel.GroupMembersIntentModel<G, I, U> {
        private final GroupIntentModel.GroupMembersIntentModel<G, I, U> gim;

        /**
         * Constructor taking user intent model.
         *
         * @param group group
         */
        public GroupMembersAspectModel(G group) {
            this.gim = intentModel.getModel(group);
        }

        /**
         * Returns an item aspect model from a list of scored items.
         *
         * @param items list of items with scores
         */
        public abstract ItemAspectModel<I, U> getItemAspectModel(List<Tuple2od<I>> items);

        @Override
        public Set<U> getIntents() {
            return gim.getIntents();
        }

        @Override
        public Stream<U> getItemIntents(I i) {
            return gim.getItemIntents(i);
        }

        @Override
        public double pu_g(U u) {
            return gim.pu_g(u);
        }
    }

    /**
     * Item aspect model for {@link AspectModel}.
     *
     * @param <I> item type
     * @param <U> aspect type
     */
    public interface ItemAspectModel<I, U> {
        /**
         * Returns probability of an item given an aspect
         *
         * @param iv item-value pair
         * @param u aspect
         * @return probability of an item given an aspect
         */
        public double pi_u(Tuple2od<I> iv, U u);
    }

}
