/**
  * Author: Sachin Joshi
  * Task 1
  * Subtask 4
  * Testing Data set
  */

import java.io.*;

class type
{
public static void main(String args[])
{
try
{
// Open the text file to be read and the text file to write the extracted attributes
BufferedReader br=new BufferedReader(new FileReader("tesj.txt")); // This text file contains the XML documents from the DrugBank corpus corresponding to testing data set
BufferedWriter br1=new BufferedWriter(new FileWriter("subst.txt", true)); // This file will contain the extracted pharmacological substance present in each sentence extracted previously
BufferedWriter br2=new BufferedWriter(new FileWriter("entity.txt", true)); // This text file will contain the entity type corresponding to each extracted pharmacological substance

String s;

//Read the entire text file line by line
while((s=br.readLine())!=null)
 {
	if(s.compareTo("    </sentence>")==0)
	{
		System.out.println(".");
		br1.write(".");
		br1.newLine();
		br2.write(".");
		br2.newLine();
	}
	for(int i=0; i<s.length(); i++)
	 {
		for(int j=i+1; j<s.length(); j++)
		 {
			
			// Search for the text phrase "type="
			if(((s.substring(i,j)).compareTo("type=")==0))
			 {
				for(int k=j+1;k<s.length(); k++)
				 {
					if(s.charAt(k)==' ')
					{
						String s2=s.substring(j+1,k-1); // Whenever the text phrase "type=" is encountered, extract the entity type
						System.out.print(s2+"\t"); // Prints the extracted entity type to the screen
						br2.write(s2); // Writes the extracted entity type to the file
					}
					if(s.charAt(k)=='=') // Continue searching for the character '='
					 {
						
						for(int z=k+2;z<s.length();z++)
						{
						  if(s.charAt(z)=='"') // When the character '=' is encountered,
							{
							String s1=s.substring(k+2,z); // Extract the entire pharmacological substance until the character '"' is encountered
							System.out.println(s1); // Prints the extracted pharmacological substance to the screen
							br1.write(s1); // Writes the extracted pharmacological substance to the file
							break;
							}
						}	
						br1.newLine(); 
						br2.newLine();
						break; 
					 }
				 }
			 }
		 }
	 }
 }//end of while loop
br.close(); //Close the text file being read 
br1.close(); //Close the text file being written
br2.close();
}//end of try block

//Catch any Input/Output Exceptions
catch(IOException e)
{
e.printStackTrace();
}

}//end of main
}//end of class

 