package experiment;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public abstract class AbstractExperimentReport {

	
	public abstract String outPutRepresentation();
	public static void saveInFile(List<AbstractExperimentReport> exps, String filePath){
		if(exps != null && exps.size() > 0){
			Path file = Paths.get(filePath);
			try(PrintWriter out = new PrintWriter(file.toFile())){
				for (AbstractExperimentReport exp : exps) {
					if(exp != null)
						out.println(exp.outPutRepresentation());
				}
			
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}
	}
}
