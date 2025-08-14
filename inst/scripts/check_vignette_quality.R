#!/usr/bin/env Rscript

# Script to check vignette quality metrics
# Run from package root: Rscript inst/scripts/check_vignette_quality.R

library(cli)

# Function to calculate readability
calc_readability <- function(text) {
  # Remove code chunks and YAML
  text <- gsub("```\\{[^}]+\\}[^`]*```", "", text)
  text <- gsub("^---\n.*?\n---\n", "", text, perl = TRUE)
  
  sentences <- length(unlist(strsplit(text, "[.!?]+")))
  words <- length(unlist(strsplit(text, "\\s+")))
  
  # Simplified syllable counting
  word_list <- unlist(strsplit(text, "\\s+"))
  syllables <- sum(sapply(word_list, function(w) {
    max(1, nchar(gsub("[^aeiouAEIOU]", "", w)))
  }))
  
  if (words > 0 && sentences > 0) {
    fk_grade <- 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    return(fk_grade)
  }
  return(NA)
}

# Check bullet usage
check_bullets <- function(file) {
  lines <- readLines(file, warn = FALSE)
  bullets <- sum(grepl("^\\s*[-*+]\\s", lines))
  
  text <- paste(lines, collapse = " ")
  text <- gsub("```\\{[^}]+\\}[^`]*```", "", text)
  words <- length(unlist(strsplit(text, "\\s+")))
  
  if (words > 0) {
    return(list(
      bullets = bullets,
      words = words,
      ratio = (bullets / words) * 100
    ))
  }
  return(list(bullets = 0, words = 0, ratio = 0))
}

# Main quality check
cli_h1("Vignette Quality Check")

vignettes <- list.files("vignettes", pattern = "\\.Rmd$", full.names = TRUE)
vignettes <- vignettes[!grepl("_template\\.Rmd$", vignettes)]

if (length(vignettes) == 0) {
  cli_alert_warning("No vignettes found")
  quit(status = 0)
}

# Initialize results
results <- data.frame(
  vignette = character(),
  grade_level = numeric(),
  word_count = numeric(),
  bullets = numeric(),
  bullet_ratio = numeric(),
  stringsAsFactors = FALSE
)

# Check each vignette
cli_h2("Analyzing vignettes...")

for (v in vignettes) {
  text <- paste(readLines(v, warn = FALSE), collapse = "\n")
  grade <- calc_readability(text)
  bullet_stats <- check_bullets(v)
  
  results <- rbind(results, data.frame(
    vignette = basename(v),
    grade_level = round(grade, 1),
    word_count = bullet_stats$words,
    bullets = bullet_stats$bullets,
    bullet_ratio = round(bullet_stats$ratio, 1),
    stringsAsFactors = FALSE
  ))
}

# Display results
cli_h2("Results")

# Readability
avg_grade <- mean(results$grade_level, na.rm = TRUE)
if (!is.na(avg_grade)) {
  if (avg_grade <= 14) {
    cli_alert_success("Average readability: {.val {round(avg_grade, 1)}} (target: ≤14) {.emph ✓}")
  } else {
    cli_alert_warning("Average readability: {.val {round(avg_grade, 1)}} (target: ≤14)")
  }
}

# Bullet usage
avg_bullets <- mean(results$bullet_ratio)
if (avg_bullets < 10) {
  cli_alert_success("Bullet usage: {.val {round(avg_bullets, 1)}} per 100 words (target: <10) {.emph ✓}")
} else {
  cli_alert_warning("Bullet usage: {.val {round(avg_bullets, 1)}} per 100 words (target: <10)")
}

# Detailed table
cli_h3("Detailed Metrics")
print(results, row.names = FALSE)

# Spell check
if (requireNamespace("spelling", quietly = TRUE)) {
  cli_h2("Spell Check")
  spell_errors <- spelling::spell_check_package()
  
  if (nrow(spell_errors) == 0) {
    cli_alert_success("No spelling errors found {.emph ✓}")
  } else {
    cli_alert_warning("Found {.val {nrow(spell_errors)}} potential spelling errors")
    print(spell_errors)
  }
}

# Summary
cli_h2("Summary")

all_good <- TRUE

if (!is.na(avg_grade) && avg_grade > 14) {
  cli_alert_info("Consider simplifying language to improve readability")
  all_good <- FALSE
}

if (avg_bullets >= 10) {
  cli_alert_info("Consider converting more bullet lists to narrative prose")
  all_good <- FALSE
}

if (any(results$word_count < 500)) {
  short_vignettes <- results$vignette[results$word_count < 500]
  cli_alert_info("Short vignettes (<500 words): {.file {short_vignettes}}")
  cli_alert_info("Consider expanding with more examples and explanations")
}

if (all_good) {
  cli_alert_success("All quality metrics passed! {.emph 🎉}")
} else {
  cli_alert_info("See suggestions above to improve vignette quality")
}

# Exit status
quit(status = ifelse(all_good, 0, 1))