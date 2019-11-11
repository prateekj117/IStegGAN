#!/usr/bin/env bash

for n in $(seq 0 12000); do mv "$n" ../image224/train; echo "$n"; done

for n in $(seq 12000 14439); do mv "$n" ../image224/test; echo "$n"; done
