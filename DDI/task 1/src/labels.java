/**
  * Author: Sachin Joshi
  * Task 1
  * Subtask 4
  */

import java.io.*;
class labels
{
	public static void main(String args[])throws IOException
	{
		try
		{
			BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
			BufferedReader br1=new BufferedReader(new FileReader("subst.txt")); // This text file contains all the pharmacological substance
			BufferedReader br2=new BufferedReader(new FileReader("entity.txt")); // This text file contains the entity type of each pharmacological substance
			BufferedReader br3=new BufferedReader(new FileReader("Words.txt")); // This text file contains all the words
			BufferedWriter br4=new BufferedWriter(new FileWriter("Labels.txt")); // This text file will contain the label of each word 
			
			int c=0, c1=0;
			String s,s1;
			String arr[]=new String[17612]; // Create a string array to store all the pharmacological substances
			String brr[]=new String[17612]; // Create a string array to store the entity type corresponding to the pharmacological substance
			String crr[]=new String[118890]; // Create a string array to store all the Words
			String drr[]=new String[118890]; // Create a string array that will store the Label of each Word
			
			// Read the text file subst.txt and entity.txt line by line
			while((s=br1.readLine())!=null&&(s1=br2.readLine())!=null)
			{
				arr[c]=s; // Store each pharmacological substance in the string array
				brr[c]=s1; // Store the entity type of each pharmaological substance in the string array
				c++;
			}
			
			br1.close();
			br2.close();
			
			//for(int i=0; i<17612; i++)
			//{
				//System.out.println(brr[i]+"\t"+arr[i]);
			//}
			
			// Read the text file Words.txt line by line
			while((s=br3.readLine())!=null)
			{
				crr[c1]=s; // Stores each word in the string array
				c1++;
			}
			
			//for(int i=0; i<118890; i++)
			//{
				//System.out.println(crr[i]);
			//}
			
			// Stores the label of each word corresponding to its entity type in a string array 
			for(int i=0; i<17612; i++)
			{
				if(arr[i]!=".")
				{
					for(int j=0; j<118890; j++)
					{
						if((arr[i].compareTo(crr[j]))==0&&(brr[i].compareTo("drug"))==0)
						{
							drr[j]="D";
						}
						if((arr[i].compareTo(crr[j]))==0&&(brr[i].compareTo("brand"))==0)
						{
							drr[j]="B";
						}
						if((arr[i].compareTo(crr[j]))==0&&(brr[i].compareTo("group"))==0)
						{
							drr[j]="G";
						}
						if((arr[i].compareTo(crr[j]))==0&&(brr[i].compareTo("drug_n"))==0)
						{
							drr[j]="D_n";
						}
						if((crr[j].compareTo("."))==0)
						{
							drr[j]=".";
						}
						if(((arr[i].compareTo(crr[j]))!=0)&&(drr[j]==null||(drr[j].compareTo("B"))!=0)&&(drr[j]==null||(drr[j].compareTo("D"))!=0)&&(drr[j]==null||(drr[j].compareTo("G"))!=0)&&(drr[j]==null||(drr[j].compareTo("D_n"))!=0)&&(drr[j]==null||(drr[j].compareTo("."))!=0))
						{
							drr[j]="O";
						}
					}	
				}
						
			}
			
			// Writes the labels of each word to the text file Labels.txt
			for(int i=0; i<118890; i++)
			{
				System.out.println(drr[i]);
				if((drr[i].compareTo("."))==0)
				{
					br4.write(drr[i]);
					br4.newLine();
					br4.newLine();
				}
				else
				{
					br4.write(drr[i]);
					br4.newLine();
				}
			}
			br4.close();
		}// end of try block
		
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}// end of main
}// end of class	