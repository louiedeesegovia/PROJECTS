%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int yylex();
void yyerror(const char *s);
extern int yylineno;
extern FILE *yyin;
extern int yynerrs; // Tracks the total number of errors

// --- Assembly & Output Generation Variables ---
FILE *data_file;
FILE *text_file;
FILE *console_file; // Buffer for standard output
int reg_counter = 1;

// --- Interpreter Symbol Table Variables ---
typedef struct {
    char *name;
    enum { TYPE_DIGIT, TYPE_LETTER, TYPE_WORD, TYPE_FLOAT } type;
    union {
        int ival;
        char cval;
        char *sval;
        double fval;
    } value;
    int initialized;
} Symbol;

Symbol symtab[100];
int symcount = 0;
int current_type = 0;

char *arg_names[10];
int arg_count = 0;

// Function prototypes
Symbol* lookup(char *name);
Symbol* install(char *name, int type);
void print_formatted(char *fmt);

%}

%error-verbose

%union {
    int ival;
    double fval;
    char cval;
    char *sval;
    struct {
        double val;
        int type;
        char *str_val;
        int reg;      /* Tracks the assembly register for this value */
    } expr_val;
}

%token NEWLINE
%token KW_DIGIT KW_LETTER KW_WORD KW_FLOAT KW_CALC KW_SHOW KW_START KW_DONE
%token COLON COMMA SEMI LPAREN RPAREN LBRACE RBRACE ARROW ASSIGN
%token PLUS MINUS STAR SLASH MOD
%token <ival> NUMBER
%token <fval> FLOAT_NUMBER
%token <cval> CHAR_LITERAL
%token <sval> FORMAT_STRING IDENTIFIER

%type <expr_val> expression term factor value_or_expr
%type <ival> type_keyword

%left PLUS MINUS
%left STAR SLASH MOD
%nonassoc UMINUS

%%

program:
    opt_newlines KW_START LBRACE statements KW_DONE RBRACE opt_newlines { YYACCEPT; }
    ;

opt_newlines:
    /* empty */
    | opt_newlines NEWLINE
    ;

statements:
    /* empty */
    | statements statement
    ;

statement:
    declaration
    | assignment
    | print_statement
    | expression_statement
    | NEWLINE
    | error NEWLINE { yyerrok; }
    ;

type_keyword:
    KW_DIGIT    { $$ = TYPE_DIGIT; current_type = TYPE_DIGIT; }
    | KW_LETTER { $$ = TYPE_LETTER; current_type = TYPE_LETTER; }
    | KW_WORD   { $$ = TYPE_WORD; current_type = TYPE_WORD; }
    | KW_FLOAT  { $$ = TYPE_FLOAT; current_type = TYPE_FLOAT; }
    ;

declaration:
    type_keyword COLON decl_list SEMI
    ;

decl_list:
    decl_item
    | decl_list COMMA decl_item
    ;

decl_item:
    IDENTIFIER {
        install($1, current_type);
        fprintf(data_file, "          %s: .byte 0\n", $1);
        
        int val_reg = reg_counter++;
        fprintf(text_file, "        daddiu r%d, r0, 0\n", val_reg);
        fprintf(text_file, "        sb r%d, %s(r0)\n", val_reg, $1);
        free($1);
    }
    | IDENTIFIER ASSIGN value_or_expr {
        Symbol *s = install($1, current_type);
        s->initialized = 1;
        
        if ($3.type == TYPE_WORD && $3.str_val != NULL) {
            s->value.sval = strdup($3.str_val);
        } else {
            s->value.fval = $3.val;
        }

        if ($3.type == TYPE_DIGIT || $3.type == TYPE_FLOAT) {
            fprintf(data_file, "          %s: .byte %d\n", $1, (int)$3.val);
        } else {
            fprintf(data_file, "          %s: .byte 0\n", $1);
        }
        
        fprintf(text_file, "        sb r%d, %s(r0)\n", $3.reg, $1);
        free($1);
    }
    ;

assignment:
    IDENTIFIER ASSIGN value_or_expr SEMI {
        Symbol *s = lookup($1);
        
        if (!s) {
            char err_msg[128];
            sprintf(err_msg, "Undeclared variable: '%s'", $1);
            yyerror(err_msg);
            free($1);
            YYABORT; 
        }
        
        s->initialized = 1;
        if ($3.type == TYPE_WORD && $3.str_val != NULL) {
            if (s->value.sval) free(s->value.sval);
            s->value.sval = strdup($3.str_val);
        } else {
            s->value.fval = $3.val;
        }
        
        if ($3.reg > 0) {
            fprintf(text_file, "        sb r%d, %s(r0)\n", $3.reg, $1);
        }
        free($1);
    }
    ;

value_or_expr:
    NUMBER {
        $$.val = (double)$1;
        $$.type = TYPE_DIGIT;
        $$.str_val = NULL;
        $$.reg = reg_counter++;
        fprintf(text_file, "        daddiu r%d, r0, %d\n", $$.reg, $1);
    }
    | MINUS NUMBER {
        $$.val = -(double)$2;
        $$.type = TYPE_DIGIT;
        $$.str_val = NULL;
        $$.reg = reg_counter++;
        fprintf(text_file, "        daddiu r%d, r0, %d\n", $$.reg, -$2);
    }
    | FLOAT_NUMBER {
        $$.val = $1;
        $$.type = TYPE_FLOAT;
        $$.str_val = NULL;
        $$.reg = reg_counter++;
        fprintf(text_file, "        daddiu r%d, r0, %d\n", $$.reg, (int)$1);
    }
    | MINUS FLOAT_NUMBER {
        $$.val = -$2;
        $$.type = TYPE_FLOAT;
        $$.str_val = NULL;
        $$.reg = reg_counter++;
        fprintf(text_file, "        daddiu r%d, r0, %d\n", $$.reg, (int)-$2);
    }
    | FORMAT_STRING {
        $$.val = 0;
        $$.type = TYPE_WORD;
        char *str = $1;
        int len = strlen(str);
        if (str[0] == '"' && str[len-1] == '"') {
            str[len-1] = '\0';
            str++;
        }
        $$.str_val = strdup(str);
        $$.reg = 0;
        free($1);
    }
    | KW_CALC expression {
        $$.val = $2.val;
        $$.type = $2.type;
        $$.str_val = NULL;
        $$.reg = $2.reg; 
    }
    ;

expression_statement:
    expression SEMI {
        fprintf(console_file, "Result: %g\n", $1.val);
    }
    ;

expression:
    expression PLUS term {
        $$.val = $1.val + $3.val;
        $$.type = ($1.type == TYPE_FLOAT || $3.type == TYPE_FLOAT) ? TYPE_FLOAT : TYPE_DIGIT;
        $$.str_val = NULL;
        $$.reg = reg_counter++; 
        fprintf(text_file, "        daddu r%d, r%d, r%d\n", $$.reg, $1.reg, $3.reg);
    }
    | expression MINUS term {
        $$.val = $1.val - $3.val;
        $$.type = ($1.type == TYPE_FLOAT || $3.type == TYPE_FLOAT) ? TYPE_FLOAT : TYPE_DIGIT;
        $$.str_val = NULL;
        $$.reg = reg_counter++;
        fprintf(text_file, "        dsubu r%d, r%d, r%d\n", $$.reg, $1.reg, $3.reg);
    }
    | term { $$ = $1; }
    ;

term:
    term STAR factor {
        $$.val = $1.val * $3.val;
        $$.type = ($1.type == TYPE_FLOAT || $3.type == TYPE_FLOAT) ? TYPE_FLOAT : TYPE_DIGIT;
        $$.str_val = NULL;
        fprintf(text_file, "        dmult r%d, r%d\n", $1.reg, $3.reg);
        $$.reg = reg_counter++;
        fprintf(text_file, "        mflo r%d\n", $$.reg);
    }
    | term SLASH factor {
        if ($3.val == 0) {
            yyerror("Division by zero");
            $$.val = 0;
        } else {
            $$.val = $1.val / $3.val;
        }
        $$.type = TYPE_FLOAT;
        $$.str_val = NULL;
        fprintf(text_file, "        ddiv r%d, r%d\n", $1.reg, $3.reg);
        $$.reg = reg_counter++;
        fprintf(text_file, "        mflo r%d\n", $$.reg);
    }
    | term MOD factor {
        if ((int)$3.val == 0) {
            yyerror("Modulo by zero");
            $$.val = 0;
        } else {
            $$.val = (int)$1.val % (int)$3.val;
        }
        $$.type = TYPE_DIGIT;
        $$.str_val = NULL;
        fprintf(text_file, "        ddiv r%d, r%d\n", $1.reg, $3.reg);
        $$.reg = reg_counter++;
        fprintf(text_file, "        mfhi r%d\n", $$.reg);
    }
    | factor { $$ = $1; }
    ;

factor:
    NUMBER {
        $$.val = (double)$1;
        $$.type = TYPE_DIGIT;
        $$.str_val = NULL;
        $$.reg = reg_counter++;
        fprintf(text_file, "        daddiu r%d, r0, %d\n", $$.reg, $1);
    }
    | FLOAT_NUMBER {
        $$.val = $1;
        $$.type = TYPE_FLOAT;
        $$.str_val = NULL;
        $$.reg = reg_counter++;
        fprintf(text_file, "        daddiu r%d, r0, %d\n", $$.reg, (int)$1);
    }
    | IDENTIFIER {
        Symbol *s = lookup($1);
        
        if (!s) {
            char err_msg[128];
            sprintf(err_msg, "Undeclared variable: '%s'", $1);
            yyerror(err_msg);
            free($1);
            YYABORT; 
        }
        
        $$.val = s->value.fval;
        $$.type = s->type;
        $$.str_val = NULL;
        
        $$.reg = reg_counter++;
        fprintf(text_file, "        lb r%d, %s(r0)\n", $$.reg, $1);
        free($1);
    }
    | MINUS factor %prec UMINUS {
        $$.val = -$2.val;
        $$.type = $2.type;
        $$.str_val = NULL;
        $$.reg = reg_counter++;
        fprintf(text_file, "        dsubu r%d, r0, r%d\n", $$.reg, $2.reg);
    }
    | LPAREN expression RPAREN { $$ = $2; }
    ;

print_statement:
    KW_SHOW ARROW LPAREN FORMAT_STRING RPAREN SEMI {
        char *str = $4;
        int len = strlen(str);
        if (str[0] == '"' && str[len-1] == '"') {
            str[len-1] = '\0';
            str++;
        }
        for (int i = 0; str[i]; i++) {
            if (str[i] == '\\' && str[i+1] == 'n') { fprintf(console_file, "\n"); i++; } 
            else if (str[i] == '\\' && str[i+1] == 't') { fprintf(console_file, "\t"); i++; } 
            else { fprintf(console_file, "%c", str[i]); }
        }
        fprintf(console_file, "\n");
        free($4);
    }
    | KW_SHOW ARROW LPAREN FORMAT_STRING COMMA { arg_count = 0; } identifier_list RPAREN SEMI {
        print_formatted($4);
        free($4);
        for (int i = 0; i < arg_count; i++) free(arg_names[i]);
    }
    ;

identifier_list:
    IDENTIFIER { 
        if (!lookup($1)) {
            char err_msg[128];
            sprintf(err_msg, "Undeclared variable: '%s'", $1);
            yyerror(err_msg);
            free($1);
            YYABORT;
        }
        arg_names[arg_count++] = strdup($1); 
        free($1); 
    }
    | identifier_list COMMA IDENTIFIER { 
        if (!lookup($3)) {
            char err_msg[128];
            sprintf(err_msg, "Undeclared variable: '%s'", $3);
            yyerror(err_msg);
            free($3);
            YYABORT;
        }
        arg_names[arg_count++] = strdup($3); 
        free($3); 
    }
    ;

%%

Symbol* lookup(char *name) {
    for (int i = 0; i < symcount; i++) {
        if (strcmp(symtab[i].name, name) == 0) return &symtab[i];
    }
    return NULL;
}

Symbol* install(char *name, int type) {
    Symbol *s = lookup(name);
    if (s != NULL) return s;
    if (symcount >= 100) return NULL;
    
    symtab[symcount].name = strdup(name);
    symtab[symcount].type = type;
    symtab[symcount].initialized = 0;
    symtab[symcount].value.sval = NULL;
    
    symcount++;
    return &symtab[symcount-1];
}

void print_formatted(char *fmt) {
    int len = strlen(fmt);
    int arg_index = 0;
    if (fmt[0] == '"' && fmt[len-1] == '"') { fmt[len-1] = '\0'; fmt++; }
    
    for (int i = 0; fmt[i]; i++) {
        if (fmt[i] == '\\' && fmt[i+1] == 'n') { fprintf(console_file, "\n"); i++; } 
        else if (fmt[i] == '\\' && fmt[i+1] == 't') { fprintf(console_file, "\t"); i++; } 
        else if (fmt[i] == '@' && fmt[i+1]) {
            i++;
            if (arg_index >= arg_count) { fprintf(console_file, "@%c", fmt[i]); continue; }
            Symbol *s = lookup(arg_names[arg_index]);
            switch (fmt[i]) {
                case 's':
                    if (s && s->type == TYPE_WORD && s->value.sval) fprintf(console_file, "%s", s->value.sval);
                    else fprintf(console_file, "(string)");
                    arg_index++; break;
                case 'd':
                    if (s && s->initialized) fprintf(console_file, "%d", (int)s->value.fval);
                    else fprintf(console_file, "0");
                    arg_index++; break;
                case 'f':
                    if (s && s->initialized) fprintf(console_file, "%g", s->value.fval);
                    else fprintf(console_file, "0");
                    arg_index++; break;
                case 'c':
                    if (s && s->type == TYPE_WORD && s->value.sval) fprintf(console_file, "%c", s->value.sval[0]);
                    else if (s && s->initialized) fprintf(console_file, "%c", (char)s->value.fval);
                    else fprintf(console_file, "?");
                    arg_index++; break;
                default: fprintf(console_file, "@%c", fmt[i]);
            }
        } else { fprintf(console_file, "%c", fmt[i]); }
    }
    fprintf(console_file, "\n");
}

int main(int argc, char **argv) {
    if (argc > 1) {
        yyin = fopen(argv[1], "r");
        if (!yyin) return 1;
    }

    data_file = fopen("data.tmp", "w");
    text_file = fopen("text.tmp", "w");
    console_file = fopen("console.tmp", "w"); // Buffer file opened
    
    fprintf(data_file, ".data\n");
    fprintf(text_file, ".code\n");

    int parse_result = yyparse();

    fclose(data_file);
    fclose(text_file);
    fclose(console_file); // Buffer file closed so we can read it

    if (parse_result == 0 && yynerrs == 0) {
        // Build final output.s
        FILE *final_out = fopen("output.s", "w");
        FILE *read_data = fopen("data.tmp", "r");
        FILE *read_text = fopen("text.tmp", "r");
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), read_data)) fputs(buffer, final_out);
        while (fgets(buffer, sizeof(buffer), read_text)) fputs(buffer, final_out);
        fclose(read_data);
        fclose(read_text);
        fclose(final_out);
        
        // Output buffered strings to the terminal
        FILE *read_console = fopen("console.tmp", "r");
        if (read_console) {
            while (fgets(buffer, sizeof(buffer), read_console)) {
                printf("%s", buffer);
            }
            fclose(read_console);
        }
        
        printf("\n\n");
        printf("Compilation Successful!\n");
    } else {
        remove("output.s"); 
    }

    remove("data.tmp");
    remove("text.tmp");
    remove("console.tmp"); // Clean up buffer

    return parse_result;
}

void yyerror(const char *s) {
    if (strstr(s, "unexpected KW_CALC") != NULL || strstr(s, "unexpected NUMBER") != NULL) {
        fprintf(stderr, "Line %d: Invalid syntax. Did you forget an '=' assignment operator?\n", yylineno);
    }
    else if (strstr(s, "expecting SEMI") != NULL) {
        fprintf(stderr, "Line %d: Missing semicolon ';' at the end of the statement.\n", yylineno);
    } 
    else if (strstr(s, "expecting RPAREN") != NULL) {
        fprintf(stderr, "Line %d: Missing closing parenthesis ')'.\n", yylineno);
    }
    else if (strstr(s, "expecting RBRACE") != NULL) {
        fprintf(stderr, "Line %d: Missing closing brace '}'.\n", yylineno);
    }
    else {
        fprintf(stderr, "Line %d: %s\n", yylineno, s);
    }
}