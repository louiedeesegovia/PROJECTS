%{
//Scanner or tokenizer for SegObC Programming Language
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "segobC.tab.h"

// Forward declaration of yyerror
void yyerror(const char *s);
%}

%option yylineno

%%

[ \t\r]+                        ; // ignore whitespace
\n                              { return NEWLINE; }

"digit"                         { return KW_DIGIT; }
"letter"                        { return KW_LETTER; }
"word"                          { return KW_WORD; }
"float"                         { return KW_FLOAT; }
"calc"                          { return KW_CALC; }
"show"                          { return KW_SHOW; }
"_start"                        { return KW_START; }
"_done"                         { return KW_DONE; }

":"                             { return COLON; }
","                             { return COMMA; }
";"                             { return SEMI; }
"("                             { return LPAREN; }
")"                             { return RPAREN; }
">>"                            { return ARROW; }
"="                             { return ASSIGN; }
"+"                             { return PLUS; }
"-"                             { return MINUS; }
"*"                             { return STAR; }
"%"				{ return MOD; }
"/"                             { return SLASH; }
"{"                             { return LBRACE; }
"}"                             { return RBRACE; }

[0-9]+\.[0-9]+                  { yylval.fval = atof(yytext); return FLOAT_NUMBER; }
\.[0-9]+                        { yylval.fval = atof(yytext); return FLOAT_NUMBER; }
[0-9]+\.                        { yylval.fval = atof(yytext); return FLOAT_NUMBER; }
[0-9]+                          { yylval.ival = atoi(yytext); return NUMBER; }

'[ -~]'                         { yylval.cval = yytext[1]; return CHAR_LITERAL; }

\"([^"\\@]|\\[nt]|\\.|@[csdf])*\"   { 
                                    yylval.sval = strdup(yytext); 
                                    return FORMAT_STRING; 
                                }

[a-zA-Z][a-zA-Z0-9_]*           { yylval.sval = strdup(yytext); return IDENTIFIER; }

.                               { fprintf(stderr, "Line %d: Invalid character '%s'\n", yylineno, yytext); }

%%

int yywrap(void) { 
    return 1; 
}
