package featureSelection.Evaluators;

import java.util.Arrays;
import java.util.BitSet;

import weka.core.Instance;
import weka.core.Utils;

public class ExpectationGeneral extends OurBaseFeatureSelectionEvaluator
{

	private int k, n;
	private double[][] cache;
	private BitSet bs;
	private Instance datum;
	
	@Override
	public double evaluateSubset( BitSet bs )
	{
		//System.out.println( Utils.bitSetToString( bs, this.dataDiscretized.numAttributes()-1 ) );
		k=bs.cardinality(); // the number of selected features. this makes n a variable as well
		n=k/2; // change n for different n-out-of-k models
		this.bs=bs;
		double score=0;
		for( int i=0 ; i<this.dataDiscretized.size() ; i++ )
		{
			Instance datum=this.dataDiscretized.get( i );
			this.datum=datum;
			cache=new double[n+1][k+1];
			for( int j=0 ; j<n ; j++ )
				Arrays.fill( cache[j], -1 );
			score+=query( 1, n, bs.nextSetBit( 0 ) );
		}
		return score;
	}
	
	private double query( int i, int j, int f )
	{
		if( j==0 )
			return 1;
		if( f==-1 || j>k-i+1 ) // TODO I feel that 1 has priority over 0...
			return 0;
		if( cache[i][j]!=-1 )
			return cache[i][j];
		double trueCase=probs[f][this.trueLabel][(int)datum.value(f)]*query( i+1, j-1, bs.nextSetBit( f+1 ) );
		double falseCase=probs[f][this.negLabel][(int)datum.value(f)]*query( i+1, j, bs.nextSetBit( f+1 ) );
		return trueCase+falseCase;
	}
	
}