package gfar.aggregation;

import es.uam.eps.ir.ranksys.core.Recommendation;

import java.util.Map;

/**
 * Interface for the Aggregation Strategies
 * @author Mesut Kaya (M.Kaya@tudelft.nl)
 *
 * @param <U> type of the users
 * @param <I> type of the items
 * @param <G> type of the groups
 */

public interface AggregationStrategy<U,I,G> {
    /**
     * Returns
     * @param recommendation recommendation list for each member of the group
     * @return a recommendation list for the group
     */
    public Recommendation<G,I> aggregate(Map<U, Recommendation<U, I>> recommendation);
}
