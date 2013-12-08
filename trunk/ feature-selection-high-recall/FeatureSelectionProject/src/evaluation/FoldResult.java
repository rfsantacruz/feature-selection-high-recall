package evaluation;

import java.util.HashMap;
import java.util.Map;

public class FoldResult {

	private Map<EClassificationMetric, Double> metricsReported;
	private String optimalSetting; 
	
	public FoldResult( ) {
		this.metricsReported = new HashMap<EClassificationMetric, Double>();
	}
	public FoldResult(String optimalString ) {
		this.metricsReported = new HashMap<EClassificationMetric, Double>();
		this.optimalSetting = optimalString;
	}
	
	public void setMetricReported(EClassificationMetric metric, double metricValue){
		this.metricsReported.put(metric, metricValue);
	}
	public double getMetricReported(EClassificationMetric metric){
		Double value = this.metricsReported.get(metric); 
		return value != null? value.doubleValue() : 0;
	}

	public String getOptimalSetting() {
		return optimalSetting;
	}

	public void setOptimalSetting(String optimalSetting) {
		this.optimalSetting = optimalSetting;
	}
	
}
