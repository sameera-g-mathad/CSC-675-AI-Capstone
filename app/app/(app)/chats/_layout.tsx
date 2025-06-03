import { Stack, useLocalSearchParams } from "expo-router";

export default function layout() {
    const params = useLocalSearchParams();
    return <Stack >
        <Stack.Screen
            name='[param]'
            options={
                {
                    headerShown: true,
                    headerTitle: '',
                }} />
    </Stack>
}