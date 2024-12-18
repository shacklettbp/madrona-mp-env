#pragma once

#include <sqlite3.h>

#include <madrona/crash.hpp>

namespace madronaMPEnv {

inline void checkSQL(sqlite3 *db, int res, const char *file, int line,
                     const char *funcname)
{
  if (res == SQLITE_OK) [[likely]] {
    return;
  }

  FATAL("DB Error: '%s' @ %s:%d in %s",
        sqlite3_errmsg(db), file, line, funcname);
}

#define REQ_SQL(db, r) ::madronaMPEnv::checkSQL(db, (r), __FILE__, __LINE__,\
                                            MADRONA_COMPILER_FUNCTION_NAME)

void execSQL(sqlite3 *db, const char *sql);

void execResetStmt(sqlite3 *db, sqlite3_stmt *stmt);

}