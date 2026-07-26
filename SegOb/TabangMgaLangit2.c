#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

#define MAX_LINE 256
#define MAX_VARS 200
#define MAX_INSTR 1000

typedef struct {
    char name[50];
    int reg;
    int memAddr;
} Variable;

typedef struct {
    char instr[256];
} Instruction;

Variable vars[MAX_VARS];
Instruction instrs[MAX_INSTR];
int varCount = 0, regCounter = 1, lineNum = 0, memCounter = 0, instrCount = 0;
FILE *outFile = NULL;

// Expression parser state
const char *exprPtr;
int validateMode = 0;

// Forward declarations for recursive descent parser
int parseExpression();
int parseTerm();
int parseFactor();
int validateExpression(const char *expr);

// ============= UTILITY FUNCTIONS =============
void trim(char *s) {
    char *start = s;
    while (isspace(*start)) start++;
    if (start != s) memmove(s, start, strlen(start) + 1);
    char *end = s + strlen(s) - 1;
    while (end >= s && isspace(*end)) *end-- = '\0';
}
int validateAndCleanStatement(char *statement) {
    char *firstSemi = strchr(statement, ';');

    /* No semicolon at all → invalid */
    if (!firstSemi) {
        return 0;
    }

    /* Remove semicolon and everything after */
    *firstSemi = '\0';
    trim(statement);

    /* ✅ If statement becomes empty, treat as valid empty statement */
    if (!statement[0]) {
        return 2;  // special code: EMPTY statement
    }

    return 1;  // valid non-empty statement
}

int isIdentifier(const char *s) {
	int i;
    if (!s || !s[0] || !(isalpha(s[0]) || s[0] == '_'))
        return 0;
    for (i = 1; s[i]; i++)
        if (!(isalnum(s[i]) || s[i] == '_'))
            return 0;
    return 1;
}

int findVar(const char *name) {
	int i;
    for (i = 0; i < varCount; i++)
        if (strcmp(vars[i].name, name) == 0)
            return i;
    return -1;
}

int getVarReg(const char *name) {
    int i = findVar(name);
    return i == -1 ? -1 : vars[i].reg;
}

int declareVar(const char *name) {
    if(validateMode){
        return 999;
    }
    int idx = findVar(name);
    if (idx != -1)
        return vars[idx].reg;
    strcpy(vars[varCount].name, name);
    vars[varCount].reg = regCounter++;
    vars[varCount].memAddr = memCounter;
    memCounter += 8;
    return vars[varCount++].reg;
}

int newTemp() {
    if(validateMode)return 999;
    return regCounter++;
}

// ============= MACHINE CODE GENERATION =============
void genMachineCode(const char *line, char *binary, char *hex) {
    int rd, rs, rt, imm;
    char varname[50];
    unsigned int code = 0;
    char type = 'U';

    if (sscanf(line, "daddiu r%d, r%d, %d", &rd, &rs, &imm) == 3) {
        code = (0x19 << 26) | (rs << 21) | (rd << 16) | (imm & 0xFFFF); type = 'I';
    } else if (sscanf(line, "daddu r%d, r%d, r%d", &rd, &rs, &rt) == 3) {
        code = (rs << 21) | (rt << 16) | (rd << 11) | 0x2D; type = 'R';
    } else if (sscanf(line, "dsubu r%d, r%d, r%d", &rd, &rs, &rt) == 3) {
        code = (rs << 21) | (rt << 16) | (rd << 11) | 0x2F; type = 'R';
    } else if (sscanf(line, "dmult r%d, r%d", &rs, &rt) == 2) {
        code = (rs << 21) | (rt << 16) | 0x1C; type = 'R';
    } else if (sscanf(line, "ddiv r%d, r%d", &rs, &rt) == 2) {
        code = (rs << 21) | (rt << 16) | 0x1E; type = 'R';
    } else if (sscanf(line, "mflo r%d", &rd) == 1) {
        code = (rd << 11) | 0x12; type = 'R';
    } else if (sscanf(line, "sb r%d, %49[^(](r0)", &rt, varname) == 2) {
        int offset = (findVar(varname) != -1) ? vars[findVar(varname)].memAddr : 0;
        code = (0x28 << 26) | (rt << 16) | (offset & 0xFFFF); type = 'I';
    } else if (sscanf(line, "lb r%d, %49[^(](r0)", &rt, varname) == 2) {
        int offset = (findVar(varname) != -1) ? vars[findVar(varname)].memAddr : 0;
        code = (0x20 << 26) | (rt << 16) | (offset & 0xFFFF); type = 'I';
    } else {
        strcpy(binary, "00000000000000000000000000000000");
        strcpy(hex, "0x00000000");
        return;
    }

    int pos = 0;
    int i;
    for (i = 31; i >= 0; i--) {
        binary[pos++] = (code & (1u << i)) ? '1' : '0';
        if ((type == 'R' && (i == 26 || i == 21 || i == 16 || i == 11 || i == 6)) ||
            (type == 'I' && (i == 26 || i == 21 || i == 16))) binary[pos++] = ' ';
    }
    binary[pos] = '\0';
    sprintf(hex, "0x%08X", code);
}
void resolveMemoryOperand(const char *in, char *out) {
    int r;
    char var[50];

    /* sb rX, var(r0) */
    if (sscanf(in, "sb r%d, %49[^(](r0)", &r, var) == 2 &&
        findVar(var) != -1) {

        int idx = findVar(var);
        sprintf(out, "sb r%d, %04d(r%d)",
                r,
                vars[idx].memAddr,
                vars[idx].reg);
        return;
    }

    /* lb rX, var(r0) */
    if (sscanf(in, "lb r%d, %49[^(](r0)", &r, var) == 2 &&
        findVar(var) != -1) {

        int idx = findVar(var);
        sprintf(out, "lb r%d, %04d(r%d)",
                r,
                vars[idx].memAddr,
                vars[idx].reg);
        return;
    }

    /* default: unchanged */
    strcpy(out, in);
}

void emit(const char *fmt, ...) {
    if (validateMode) return;

    char raw[256];
    char resolved[256];
    va_list args;

    va_start(args, fmt);
    vsnprintf(raw, sizeof(raw), fmt, args);
    va_end(args);

    int len = strlen(raw);
    if (len > 0 && raw[len - 1] == '\n')
        raw[len - 1] = '\0';

    /* Resolve memory operands for display */
    resolveMemoryOperand(raw, resolved);

    /* Machine code must use ORIGINAL form */
    char binary[100], hex[20];
    genMachineCode(raw, binary, hex);

    /* Terminal output */
    printf("%-35s %-45s %s\n", resolved, binary, hex);

    /* Store resolved instruction for file output */
    if (instrCount < MAX_INSTR)
        strcpy(instrs[instrCount++].instr, resolved);
}


// ============= RECURSIVE DESCENT PARSER =============
void skipSpaces() {
    while (*exprPtr && isspace(*exprPtr)) exprPtr++;
}
int validateExpression(const char *expr){
    validateMode = 1;
    int savedRegCounter = regCounter;
    exprPtr = expr;
    int result = parseExpression();
    validateMode = 0;
    regCounter = savedRegCounter;
    return(result != -1 && *exprPtr == '\0')? 1 : 0;
}
int parseFactor() {
    skipSpaces();

    // Handle parentheses: ( expression )
    if (*exprPtr == '(') {
        exprPtr++;
        int result = parseExpression();
        skipSpaces();
        if (*exprPtr == ')') exprPtr++;
        return result;
    }

    // Handle numbers
    if (isdigit(*exprPtr)) {
        int value = 0;
        while (isdigit(*exprPtr)) value = value * 10 + (*exprPtr++ - '0');
        if(isalpha(*exprPtr) || *exprPtr == '_'){
            return -1;
        }
        int reg = newTemp();
        emit("daddiu r%d, r0, %d\n", reg, value);
        return reg;
    }

  // Handle identifiers (variables)
if (isalpha(*exprPtr) || *exprPtr == '_') {
    char name[64] = {0};
    int i = 0;
    while (isalnum(*exprPtr) || *exprPtr == '_')
        name[i++] = *exprPtr++;
    name[i] = '\0';

    // ❗ Variable must already exist
    if (findVar(name) == -1) {
        return -1;  // undeclared variable → error
    }

    int reg = newTemp();
    emit("lb r%d, %s(r0)\n", reg, name);
    return reg;
}
    return -1;
}

int parseTerm() {
    int left = parseFactor();
    if (left == -1)
        return -1;

    skipSpaces();
    while (*exprPtr == '*' || *exprPtr == '/') {
        char op = *exprPtr++;
        int right = parseFactor();
        if (right == -1)
            return -1;

        int result = newTemp();
        if (op == '*') {
            emit("dmult r%d, r%d\n", left, right);
            emit("mflo r%d\n", result);
        } else {
            emit("ddiv r%d, r%d\n", left, right);
            emit("mflo r%d\n", result);
        }
        left = result;
        skipSpaces();
    }

    return left;
}

int parseExpression() {
    int left = parseTerm();
    if (left == -1)
        return -1;

    skipSpaces();
    while (*exprPtr == '+' || *exprPtr == '-') {
        char op = *exprPtr++;
        int right = parseTerm();
        if (right == -1)
            return -1;

        int result = newTemp();
        if (op == '+') emit("daddu r%d, r%d, r%d\n", result, left, right);
        else emit("dsubu r%d, r%d, r%d\n", result, left, right);

        left = result;
        skipSpaces();
    }

    return left;
}

// ============= STATEMENT HANDLERS =============
void handleAssignment(const char *line) {
    char copy[MAX_LINE];
    strcpy(copy, line);
    trim(copy);

    char varname[64];
    char expr[MAX_LINE];

    int len = strlen(copy);

    /* ===== POSTFIX b++ ===== */
    if (len > 2 && copy[len-2] == '+' && copy[len-1] == '+') {
        copy[len-2] = '\0';
        strcpy(varname, copy);
        trim(varname);

        if (!isIdentifier(varname) || getVarReg(varname) == -1) {
            printf("Error in line %d\n", lineNum);
            return;
        }

        snprintf(expr, sizeof(expr), "%s + 1", varname);
        exprPtr = expr;
        int result = parseExpression();
        emit("sb r%d, %s(r0)\n", result, varname);
        return;
    }

    /* ===== POSTFIX b-- ===== */
    if (len > 2 && copy[len-2] == '-' && copy[len-1] == '-') {
        copy[len-2] = '\0';
        strcpy(varname, copy);
        trim(varname);

        if (!isIdentifier(varname) || getVarReg(varname) == -1) {
            printf("Error in line %d\n", lineNum);
            return;
        }

        snprintf(expr, sizeof(expr), "%s - 1", varname);
        exprPtr = expr;
        int result = parseExpression();
        emit("sb r%d, %s(r0)\n", result, varname);
        return;
    }

    /* ===== PREFIX ++b ===== */
    if (len > 2 && copy[0] == '+' && copy[1] == '+') {
        strcpy(varname, copy + 2);
        trim(varname);

        if (!isIdentifier(varname) || getVarReg(varname) == -1) {
            printf("Error in line %d\n", lineNum);
            return;
        }

        snprintf(expr, sizeof(expr), "%s + 1", varname);
        exprPtr = expr;
        int result = parseExpression();
        emit("sb r%d, %s(r0)\n", result, varname);
        return;
    }

    /* ===== PREFIX --b ===== */
    if (len > 2 && copy[0] == '-' && copy[1] == '-') {
        strcpy(varname, copy + 2);
        trim(varname);

        if (!isIdentifier(varname) || getVarReg(varname) == -1) {
            printf("Error in line %d\n", lineNum);
            return;
        }

        snprintf(expr, sizeof(expr), "%s - 1", varname);
        exprPtr = expr;
        int result = parseExpression();
        emit("sb r%d, %s(r0)\n", result, varname);
        return;
    }

    /* ===== COMPOUND += ===== */
    char *plusEq = strstr(copy, "+=");
    if (plusEq) {
        *plusEq = '\0';
        strcpy(varname, copy);
        trim(varname);

        strcpy(expr, plusEq + 2);
        trim(expr);

        if (!isIdentifier(varname) || getVarReg(varname) == -1) {
            printf("Error in line %d\n", lineNum);
            return;
        }

        char rewritten[MAX_LINE];
        snprintf(rewritten, sizeof(rewritten), "%s + (%s)", varname, expr);

        if (!validateExpression(rewritten)) {
            printf("Error in line %d\n", lineNum);
            return;
        }

        exprPtr = rewritten;
        int result = parseExpression();
        emit("sb r%d, %s(r0)\n", result, varname);
        return;
    }

    /* ===== NORMAL = ===== */
    char *eq = strchr(copy, '=');
    if (!eq)
        return;

    *eq = '\0';
    strcpy(varname, copy);
    trim(varname);

    strcpy(expr, eq + 1);
    trim(expr);

    if (!isIdentifier(varname) || getVarReg(varname) == -1) {
        printf("Error in line %d\n", lineNum);
        return;
    }

    if (!validateExpression(expr)) {
        printf("Error in line %d\n", lineNum);
        return;
    }

    exprPtr = expr;
    int result = parseExpression();
    emit("sb r%d, %s(r0)\n", result, varname);
}


void handleDeclaration(const char *line) {
	char *p;
    if (strncmp(line, "int", 3) != 0)
        return;
    char buf[MAX_LINE];
    strcpy(buf, line + 3);
    trim(buf);
    if (!buf[0])
        return;
    for (p = buf; *p; ) {
        char item[MAX_LINE] = {0};
        int j = 0;
        while (*p && *p != ',') item[j++] = *p++;
        if (*p == ',') p++;
        trim(item);
        if (!item[0])
            continue;
        char *eq = strchr(item, '=');
        if (eq) {
            *eq = '\0';
            char varname[64], expr[MAX_LINE];
            strcpy(varname, item);
            trim(varname);
            strcpy(expr, eq + 1);
            trim(expr);
            if (expr[strlen(expr) - 1] == ';') expr[strlen(expr) - 1] = '\0';
            trim(expr);

            if (!isIdentifier(varname) || findVar(varname) != -1) {
                printf("Error in line %d\n", lineNum);
                continue;
            }
            if(!validateExpression(expr)){
                printf("Error in line %d\n", lineNum);
                continue;
            }
            // Parse and generate code for expression
            exprPtr = expr;
            int result = parseExpression();

            if (result == -1) {
                printf("Error in line %d\n", lineNum);
                continue;
            }

            // Register variable with result register
            strcpy(vars[varCount].name, varname);
            vars[varCount].reg = result;
            vars[varCount].memAddr = memCounter;
            memCounter += 8;
            varCount++;

            emit("sb r%d, %s(r0)\n", result, varname);
        } else {
            char varname[64];
            strcpy(varname, item);
            trim(varname);
            if (varname[strlen(varname) - 1] == ';') varname[strlen(varname) - 1] = '\0';
            trim(varname);
            if (!isIdentifier(varname) || findVar(varname) != -1) {
                printf("Error in line %d\n", lineNum);
                continue;
            }
            int reg = regCounter++;
            strcpy(vars[varCount].name, varname);
            vars[varCount].reg = reg;                 // ✅ correct register
            vars[varCount].memAddr = memCounter;      // ✅ assign offset
            memCounter += 8;                          // ✅ advance offset
            varCount++;

            emit("daddiu r%d, r0, 0\n", reg);
            emit("sb r%d, %s(r0)\n", reg, varname);

        }
    }
}

void writeOutput() {
    if (!outFile)
        return;

    fprintf(outFile, ".data\n");
    int i;
    for (i = 0; i < varCount; i++)
        fprintf(outFile, "          %s: .byte 0\n", vars[i].name);

    fprintf(outFile, ".code\n");
	
    for(i = 0; i < instrCount; i++) {
        int rd, rt;
        char vname[50];

        /* STORE: sb rX, var(r0)  →  sb rX, offset(index) */
        if (sscanf(instrs[i].instr, "sb r%d, %49[^(](r0)", &rt, vname) == 2
            && findVar(vname) != -1) {

            int idx = findVar(vname);
            int indexReg = vars[idx].reg;
            int offset   = vars[idx].memAddr;

            fprintf(outFile,
                "        sb r%d, %04d(r%d)\n",
                rt, offset, indexReg);
        }

        /* LOAD: lb rX, var(r0)  →  lb rX, offset(index) */
        else if (sscanf(instrs[i].instr, "lb r%d, %49[^(](r0)", &rd, vname) == 2
                 && findVar(vname) != -1) {

            int idx = findVar(vname);
            int indexReg = vars[idx].reg;
            int offset   = vars[idx].memAddr;

            fprintf(outFile,
                "        lb r%d, %04d(r%d)\n",
                rd, offset, indexReg);
        }

        /* All other instructions unchanged */
        else {
            fprintf(outFile, "        %s\n", instrs[i].instr);
        }
    }
}


// ============= MAIN =============
int main(void) {
    FILE *fp = fopen("input.txt", "r");
    if (!fp) {
        printf("Error: cannot open input.txt\n");
        return 1;
    }

    outFile = fopen("AssemblyCode.txt", "w");
    if (!outFile) {
        fprintf(stderr, "Error: cannot create output.txt\n");
        fclose(fp);
        return 1;
    }

    char line[MAX_LINE];
    while (fgets(line, sizeof(line), fp)) {
    lineNum++;
    trim(line);
    if (!line[0])
        continue;

    /* Remove multiple consecutive semicolons */
    char cleanedLine[MAX_LINE];
    int j = 0;
    int lastWasSemi = 0;
	int i;
	
    for (i = 0; line[i]; i++) {
        if (line[i] == ';') {
            if (!lastWasSemi) {
                cleanedLine[j++] = line[i];
                lastWasSemi = 1;
            }
        } else {
            cleanedLine[j++] = line[i];
            lastWasSemi = 0;
        }
    }
    cleanedLine[j] = '\0';
    strcpy(line, cleanedLine);

    /* Print line only if it contains something other than semicolons */
        if (strspn(line, ";") != strlen(line)) {
            printf("%s\n", line);
        }


    /* Split into statements */
    char *stmt;
    for (stmt = line; ; ) {

        char *semi = strchr(stmt, ';');
        char statement[MAX_LINE];

        if (semi) {
            int len = semi - stmt + 1;
            strncpy(statement, stmt, len);
            statement[len] = '\0';
            stmt = semi + 1;
        } else {
            strcpy(statement, stmt);
            stmt = NULL;
        }

        trim(statement);

        /* ✅ Ignore empty statements */
        if (!statement[0]) {
            if (!stmt) break;
            continue;
        }

       int v = validateAndCleanStatement(statement);

/* ❌ invalid */
if (v == 0) {
    printf("Error in line %d\n", lineNum);
    if (!stmt) break;
    continue;
}

/* ✅ empty statement → ignore */
if (v == 2) {
    if (!stmt) break;
    continue;
}


        /* Dispatch statement */
        if (strncmp(statement, "int ", 4) == 0) {
    handleDeclaration(statement);
}
/* ✅ increment / decrement statements */
else if (strstr(statement, "++") || strstr(statement, "--")) {
    handleAssignment(statement);
}
/* normal assignment */
else if (strchr(statement, '=')) {
    handleAssignment(statement);
}
/* pure expression */
else {
    char expr[MAX_LINE];
    strcpy(expr, statement);

    if(!validateExpression(expr)){
        printf("Error in line %d\n", lineNum);
    } else {
        exprPtr = expr;
        parseExpression();
    }
}

        if (!stmt)
            break;
    }

    printf("\n");
}


    fclose(fp);
    writeOutput();
    fclose(outFile);
    printf("Assembly code written to AssemblyCode.txt\n");
    return 0;
}
