#include <list>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
using namespace std;

static void
replace(string & s, const string & s1, const string & s2, const char skip = 0);

void
tokenize(const string & s1, list<string> & lt)
{
  lt.clear();
    
  string s(s1);
  
  if (s[0] == '"') s.replace(0, 1, "`` ");
  replace(s, " \"", "  `` ");
  replace(s, "(\"", "( `` ");
  replace(s, "[\"", "[ `` ");
  replace(s, "{\"", "{ `` ");
  replace(s, "<\"", "< `` ");

  replace(s, "...", " ... ");

  replace(s, ",", " , ");
  replace(s, ";", " ; ");
  replace(s, ":", " : ");
  replace(s, "@", " @ ");
  replace(s, "#", " # ");
  replace(s, "$", " $ ");
  replace(s, "%", " % ");
  replace(s, "&", " & ");

  int pos = s.size() - 1;
  while (pos > 0 && s[pos] == ' ') pos--;
  while (pos > 0) {
    char c = s[pos];
    if (c == '[' || c == ']' || c == ')' || c == '}' || c == '>' ||
        c == '"' || c == '\'') {
      pos--; continue;
    }
    break;
  }
  if (pos >= 0 && s[pos] == '.' && !(pos > 0 && s[pos-1] == '.')) s.replace(pos, 1, " .");
  
  replace(s, "?", " ? ");
  replace(s, "!", " ! ");
    
  replace(s, "[", " [ ");
  replace(s, "]", " ] ");
  replace(s, "(", " ( ");
  replace(s, ")", " ) ");
  replace(s, "{", " { ");
  replace(s, "}", " } ");
  replace(s, "<", " < ");
  replace(s, ">", " > ");

  replace(s, "--", " -- ");

  s.replace(string::size_type(0), 0, " ");
  s.replace(s.size(), 0, " ");
  
  replace(s, "\"", " '' ");

  replace(s, "' ", " ' ", '\'');
  replace(s, "'s ", " 's ");
  replace(s, "'S ", " 'S ");
  replace(s, "'m ", " 'm ");
  replace(s, "'M ", " 'M ");
  replace(s, "'d ", " 'd ");
  replace(s, "'D ", " 'D ");
  replace(s, "'ll ", " 'll ");
  replace(s, "'re ", " 're ");
  replace(s, "'ve ", " 've ");
  replace(s, "n't ", " n't ");
  replace(s, "'LL ", " 'LL ");
  replace(s, "'RE ", " 'RE ");
  replace(s, "'VE ", " 'VE ");
  replace(s, "N'T ", " N'T ");

  replace(s, " Cannot ", " Can not ");
  replace(s, " cannot ", " can not ");
  replace(s, " D'ye ", " D' ye ");
  replace(s, " d'ye ", " d' ye ");
  replace(s, " Gimme ", " Gim me ");
  replace(s, " gimme ", " gim me ");
  replace(s, " Gonna ", " Gon na ");
  replace(s, " gonna ", " gon na ");
  replace(s, " Gotta ", " Got ta ");
  replace(s, " gotta ", " got ta ");
  replace(s, " Lemme ", " Lem me ");
  replace(s, " lemme ", " lem me ");
  replace(s, " More'n ", " More 'n ");
  replace(s, " more'n ", " more 'n ");
  replace(s, "'Tis ", " 'T is ");
  replace(s, "'tis ", " 't is ");
  replace(s, "'Twas ", " 'T was ");
  replace(s, "'twas ", " 't was ");
  replace(s, " Wanna ", " Wan na ");
  replace(s, " wanna ", " wanna ");

  istringstream is(s);
  string t;
  while (is >> t) {
    lt.push_back(t);
  }

}

static void
replace(string & s, const string & s1, const string & s2, const char skip)
{
  string::size_type pos = 0;
  while (1) {
    string::size_type i = s.find(s1, pos);
    if (i == string::npos) break;
    if (i > 0 && s[i-1] == skip) {
      pos = i + 1;
      continue;
    }
    s.replace(i, s1.size(), s2);
    pos = i + s2.size();
  }

}

int main(int argc, char** argv) {
  string inFile = "";
  string outFile = "";
  if( argc == 3 ) {
    inFile = argv[1];
    outFile = argv[2];
  }
  else {
    cout << "Usage: ./cppfile InputFile OutputFile\n";
    return 1;
  }

  string geniass_output_file = "output-geniass";
  string sentence_count_file = "sentence-count";
  string command = "./geniass " + inFile + " " + geniass_output_file;
  system(command.c_str());

  list<string> l;
  std::ifstream input( geniass_output_file.c_str() );
  ofstream out(outFile.c_str());
  ofstream sen_out(sentence_count_file.c_str(), std::fstream::out | std::fstream::app);

  //std::stringstream buffer;
  //buffer << input.rdbuf();
  //const string s = buffer.str();
  //tokenize(s, l);
  //list<string>::const_iterator i;
  //for( i = l.begin(); i != l.end(); ++i)
  //  cout << *i << "\t";
  //out << *i << " ";
  std::locale loc;
  int count = 0;
  for( std::string line; getline( input, line ); )
  {
	count++;
        string s1 = line;
        //while ( s1.back() != '.')
        //{
        //  s1.pop_back();
        //}
        //s1.pop_back();
        const string s = s1;
        tokenize(s, l);
        list<string>::const_iterator i;
        for( i = l.begin(); i != l.end(); ++i)
          out << *i << " ";
          //cout << *i << "\t";
  }
  input.close();
  out.close();
  sen_out << count << "\n";
  sen_out.close();
  string del_command = "rm " + geniass_output_file;
  system(del_command.c_str());
  return 0;
}
