# THIS SESSION

0a. Study @SPECS.md thoroughly
0b. Study @implementation_plan.md carefully
0c. Because the Phases in @implementation_plan.md are sequential, it's important to do them in order
0d. Some phases (esp phase 1) may require connection to the local TimeScale database, the credentials for which can be found in ./.env
    0di. Any step that requires MCP access to the database can instead be re-interpreted to simply connect to the db directly, using the credentials in this .env file
    0dii. Any existing code that relies on MCP should be refactored to simply directly query the db
0e. Pick one task to implement. Execute a handoff when the task is done: update SPECS.md and stop the session
0f. A file is finished when i) it has total type coverage, ii) there are zero errors and warnings, and iii) QA is done.
0g. Don't start new files if there are unfinished files.
0h. Don't start new apps/modules if there are unfinished apps/modules.
0i. If you finish a file in your session, use /commit
0j. QA feedback should usually be implemented
0k. If all four phases are marked as complete in SPECS.md, you are already done and can stop your session early.

# Updating SPECS.md:

1a. Use DRY documentation in SPECS.md
1b. Keep track of files and their error counts
1c. Keep track of progress on each file
1d. Record type design best practices you learn

# Getting feedback

2a. Run type checking with uv
2a. Write an unbiased unit test to verify behavior.
