# Stage 6: Output Format

## LLM Output Format (Simplified)

The LLM returns individual theory assignments in a simple JSON format:

```json
{
  "theory_assignments": [
    {
      "theory_id": "T000001",
      "subcluster_name": "ROS-Induced Cellular Senescence Theory"
    },
    {
      "theory_id": "T000002",
      "subcluster_name": "ROS-Induced Cellular Senescence Theory"
    },
    {
      "theory_id": "T000010",
      "subcluster_name": "Telomere-Associated Cellular Senescence Theory"
    },
    ...
  ]
}
```

### Key Features

- **Individual assignments**: Each theory_id is explicitly assigned to a subcluster
- **Simple structure**: Just theory_id and subcluster_name
- **No grouping**: LLM doesn't need to count or group theories
- **Easy to validate**: Check all theory_ids are present and no duplicates

## Internal Processing

The system automatically converts this to grouped format for internal use:

```json
{
  "subclusters": [
    {
      "subcluster_name": "ROS-Induced Cellular Senescence Theory",
      "theory_ids": ["T000001", "T000002", ...],
      "theory_count": 245,
      "mechanism_focus": ""
    },
    {
      "subcluster_name": "Telomere-Associated Cellular Senescence Theory",
      "theory_ids": ["T000010", "T000011", ...],
      "theory_count": 312,
      "mechanism_focus": ""
    }
  ],
  "separation_rationale": "Separated based on mechanistic themes",
  "original_format": {
    "theory_assignments": [...]
  }
}
```

## Example: Separating "Cellular Senescence Theory"

### Input to LLM (Batch 1 of 43)

```
# THEORIES TO SEPARATE

## Theory 1 (ID: T000040)
Original Name: Cellular Senescence Theory
Paper: Senescence and aging: causes, consequences, and therapeutic avenues
Key Concepts:
  • ROS accumulation: Reactive oxygen species accumulate with age...
  • Mitochondrial dysfunction: Damaged mitochondria produce more ROS...

## Theory 2 (ID: T000075)
Original Name: Replicative Senescence Theory
Paper: Telomere shortening triggers senescence
Key Concepts:
  • Telomere attrition: Progressive shortening of telomeres...
  • DNA damage response: Short telomeres activate DDR...

... (28 more theories)
```

### LLM Output

```json
{
  "theory_assignments": [
    {
      "theory_id": "T000040",
      "subcluster_name": "ROS-Induced Cellular Senescence Theory"
    },
    {
      "theory_id": "T000075",
      "subcluster_name": "Telomere-Associated Cellular Senescence Theory"
    },
    ...
  ]
}
```

### After Conversion (Internal Format)

```json
{
  "subclusters": [
    {
      "subcluster_name": "ROS-Induced Cellular Senescence Theory",
      "theory_ids": ["T000040", "T000123", ...],
      "theory_count": 8
    },
    {
      "subcluster_name": "Telomere-Associated Cellular Senescence Theory",
      "theory_ids": ["T000075", "T000156", ...],
      "theory_count": 12
    },
    {
      "subcluster_name": "p53-Mediated Cellular Senescence Theory",
      "theory_ids": ["T000089", "T000234", ...],
      "theory_count": 10
    }
  ]
}
```

## Validation Rules

The system validates:

1. ✅ All input theory_ids are assigned
2. ✅ No theory_id appears twice
3. ✅ Each subcluster has ≥5 theories (configurable)
4. ✅ Subcluster names are more specific than original cluster name

If validation fails, the **entire batch** is retried (up to 3 attempts).

## Fallback: Singleton Warning

If a batch fails after all retries, theories are assigned to the **original cluster name** with a `singleton_warning` status:

```json
{
  "subcluster_name": "Cellular Senescence Theory",
  "theory_ids": ["T000040", "T000075", ...],
  "theory_count": 30,
  "mechanism_focus": "",
  "status": "singleton_warning",
  "warning_reason": "Failed to separate after 3 retries"
}
```

This ensures:
- ✅ No theories are lost
- ✅ Failed batches are clearly marked
- ✅ Can be manually reviewed later
- ✅ Processing continues for other batches

## Benefits of This Format

### ✅ Simpler for LLM
- No need to count theories
- No need to create summary statistics
- Just assign each theory to a subcluster

### ✅ Easier to Validate
- Clear one-to-one mapping
- Easy to check completeness
- Easy to detect duplicates

### ✅ More Flexible
- Can add additional fields later if needed
- Easy to parse and process
- Clear data structure

### ✅ Preserves Individual Assignments
- Original assignments stored in `original_format`
- Can trace back which theory went where
- Useful for debugging and analysis
