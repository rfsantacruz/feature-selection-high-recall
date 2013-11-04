package featureSelection;

public enum EFeatureSelectionAlgorithm {

	//wrappers ******
	
	FORWARD_SELECTION_WRAPPER,
	//wrapper foward search technique
	
	BACKWARD_SELECTION_WRAPPER(),
	//wrapper BACKWARD search technique
	
	//ranks **********
	
	CONDITIONAL_ENTROPY_RANK,
	//rank features by conditional entropy

	CORRELATION_BASED_RANK,
	//Evaluates the worth of an attribute by measuring the correlation (Pearson's) between it and the class.
	//LINK: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/CorrelationAttributeEval.html

	GAINRATIO_RANK,
	//Evaluates the worth of an attribute by measuring the gain ratio with respect to the class.
	//GainR(Class, Attribute) = (H(Class) - H(Class | Attribute)) / H(Attribute).
	//link: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/GainRatioAttributeEval.html

	INFORMATIONGAIN_RANK,
	//Rank by mutual information between class and feature
	//InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).
	//link: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/InfoGainAttributeEval.html

	SYMMETRICAL_UNCERT_RANK,
	//SymmU(Class, Attribute) = 2 * (H(Class) - H(Class | Attribute)) / H(Class) + H(Attribute).
	//LINK: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/SymmetricalUncertAttributeEval.html
	
	//subset ************
	
	CORRELATION_BASED_SUBSET,
	//Subsets of features that are highly correlated with the class while having low intercorrelation are preferred
	//link: http://weka.sourceforge.net/doc.dev/weka/attributeSelection/CfsSubsetEval.html
	
	MRMR_MI_BASED_SUBSET
	//Minimum-redundancy-maximum-relevance using mi
	//LINK: http://en.wikipedia.org/wiki/Feature_selection#Minimum-redundancy-maximum-relevance_.28mRMR.29_feature_selection

	
}
